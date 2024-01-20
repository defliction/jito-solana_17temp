use log::warn;
use solana_client::rpc_client::SerializableTransaction;
use solana_metrics::datapoint_info;
use solana_rpc_client_api::config::RpcSendTransactionConfig;
use solana_sdk::commitment_config::CommitmentLevel;
use solana_sdk::signature::Signature;
use tokio::time::sleep;
use {
    crate::GeneratedMerkleTreeCollection,
    anchor_lang::{AccountDeserialize, InstructionData, ToAccountMetas},
    itertools::Itertools,
    jito_tip_distribution::state::{ClaimStatus, Config, TipDistributionAccount},
    log::{error, info},
    solana_client::nonblocking::rpc_client::RpcClient,
    solana_program::{
        fee_calculator::DEFAULT_TARGET_LAMPORTS_PER_SIGNATURE, native_token::LAMPORTS_PER_SOL,
        system_program,
    },
    solana_sdk::{
        account::Account,
        commitment_config::CommitmentConfig,
        compute_budget::ComputeBudgetInstruction,
        instruction::Instruction,
        pubkey::Pubkey,
        signature::{Keypair, Signer},
        transaction::Transaction,
    },
    std::{
        collections::HashMap,
        sync::Arc,
        time::{Duration, Instant},
    },
    thiserror::Error,
};

#[derive(Error, Debug)]
pub enum ClaimMevError {
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    JsonError(#[from] serde_json::Error),

    #[error(transparent)]
    AnchorError(anchor_lang::error::Error),

    #[error("TDA not found for pubkey: {0:?}")]
    TDANotFound(Pubkey),

    #[error("Claim Status not found for pubkey: {0:?}")]
    ClaimStatusNotFound(Pubkey),

    #[error("Claimant not found for pubkey: {0:?}")]
    ClaimantNotFound(Pubkey),

    #[error(transparent)]
    RpcError(#[from] solana_rpc_client_api::client_error::Error),

    #[error("Failed after {attempts} retries. {remaining_transaction_count} remaining mev claim transactions, {failed_transaction_count} failed requests.",)]
    MaxSendTransactionRetriesExceeded {
        attempts: u64,
        remaining_transaction_count: usize,
        failed_transaction_count: usize,
    },

    #[error("Expected to have at least {desired_balance} lamports in {payer:?}. Current balance is {start_balance} lamports. Deposit {sol_to_deposit} SOL to continue.")]
    InsufficientBalance {
        desired_balance: u64,
        payer: Pubkey,
        start_balance: u64,
        sol_to_deposit: u64,
    },
}

/// Returns a list of claim transactions for valid, unclaimed MEV tips
/// A valid unclaim transaction consist of the following:
/// - the claimant (typically a stake account) must exist.
/// - the claimant must have enough lamports post-claim to be rent-exempt
/// - it must not have already been claimed
pub async fn get_claim_transactions_for_valid_unclaimed(
    rpc_client: &RpcClient,
    merkle_trees: &GeneratedMerkleTreeCollection,
    tip_distribution_program_id: Pubkey,
    micro_lamports_per_compute_unit: u64,
    payer_pubkey: Pubkey,
) -> Result<Vec<Transaction>, ClaimMevError> {
    let tree_nodes = merkle_trees
        .generated_merkle_trees
        .iter()
        .flat_map(|tree| &tree.tree_nodes)
        .collect_vec();

    info!(
        "reading tip distribution accounts for epoch {}",
        merkle_trees.epoch
    );

    let tda_pubkeys = merkle_trees
        .generated_merkle_trees
        .iter()
        .map(|tree| tree.tip_distribution_account)
        .collect_vec();
    let tdas = crate::get_batched_accounts(&rpc_client, &tda_pubkeys).await?;

    let claimant_pubkeys = tree_nodes
        .iter()
        .map(|tree_node| tree_node.claimant)
        .collect_vec();
    let claimants = crate::get_batched_accounts(&rpc_client, &claimant_pubkeys).await?;

    let claim_status_pubkeys = tree_nodes
        .iter()
        .map(|tree_node| tree_node.claim_status_pubkey)
        .collect_vec();
    let claim_statuses = crate::get_batched_accounts(&rpc_client, &claim_status_pubkeys).await?;

    // can be helpful for determining mismatch in state between requested and read
    let tdas_onchain = tdas.values().filter(|a| a.is_some()).count();
    let claimants_onchain = claimants.values().filter(|a| a.is_some()).count();
    let claim_statuses_onchain = claim_statuses.values().filter(|a| a.is_some()).count();
    datapoint_info!(
        "get_claim_transactions_for_valid_unclaimed",
        ("tdas", tda_pubkeys.len(), i64),
        ("tdas_onchain", tdas_onchain, i64),
        ("claimants", claimant_pubkeys.len(), i64),
        ("claimants_onchain", claimants_onchain, i64),
        ("claim_statuses", claim_status_pubkeys.len(), i64),
        ("claim_statuses_onchain", claim_statuses_onchain, i64),
    );

    let transactions = build_transactions_new(
        tip_distribution_program_id,
        &merkle_trees,
        tdas,
        claimants,
        claim_statuses,
        micro_lamports_per_compute_unit,
        payer_pubkey,
    );

    Ok(transactions)
}

pub async fn claim_mev_tips(
    merkle_trees: GeneratedMerkleTreeCollection,
    rpc_url: String,
    tip_distribution_program_id: Pubkey,
    keypair: Arc<Keypair>,
    max_loop_duration: Duration,
    micro_lamports_per_compute_unit: u64,
) -> Result<(), ClaimMevError> {
    let rpc_client = RpcClient::new_with_timeout_and_commitment(
        rpc_url,
        Duration::from_secs(300),
        CommitmentConfig::confirmed(),
    );

    let max_loop_iteration_time = max_loop_duration.div_f32(10.0);

    let start = Instant::now();
    while start.elapsed() <= max_loop_duration {
        let claim_transactions = get_claim_transactions_for_valid_unclaimed(
            &rpc_client,
            &merkle_trees,
            tip_distribution_program_id,
            micro_lamports_per_compute_unit,
            keypair.pubkey(),
        )
        .await?;

        let loop_time =
            max_loop_iteration_time.min(max_loop_duration.saturating_sub(start.elapsed()));
        info!(
            "running send loop for {}s to send {} transactions",
            loop_time.as_secs_f32(),
            claim_transactions.len()
        );

        let blockhash = rpc_client.get_latest_blockhash().await?;
        let mut claim_transactions: HashMap<Signature, Transaction> = claim_transactions
            .into_iter()
            .map(|mut tx| {
                tx.sign(&[&keypair], blockhash);
                (*tx.get_signature(), tx)
            })
            .collect();

        let send_start = Instant::now();
        while send_start.elapsed() < loop_time {
            let mut num_send_ok: usize = 0;
            let mut num_send_error: usize = 0;
            let mut num_landed_ok: usize = 0;
            let mut num_landed_error: usize = 0;

            // TODO (LB): need to randomize claim_transactions and maybe send a smaller subset per loop
            while rpc_client
                .is_blockhash_valid(&blockhash, CommitmentConfig::processed())
                .await?
            {
                info!("sending {} transactions", claim_transactions.len());
                for tx in claim_transactions.values() {
                    match rpc_client
                        .send_transaction_with_config(
                            tx,
                            RpcSendTransactionConfig {
                                skip_preflight: false,
                                preflight_commitment: Some(CommitmentLevel::Confirmed),
                                max_retries: Some(2),
                                ..RpcSendTransactionConfig::default()
                            },
                        )
                        .await
                    {
                        Ok(_) => {
                            num_send_ok = num_send_ok.saturating_add(1);
                        }
                        Err(e) => {
                            warn!("error sending transaction: {:?}", e);
                            num_send_error = num_send_error.saturating_add(1);
                        }
                    }
                }

                sleep(Duration::from_secs(10)).await;

                let signatures = claim_transactions.keys().cloned().collect_vec();
                match rpc_client.get_signature_statuses(&signatures).await {
                    Ok(statuses) => {
                        signatures.iter().zip(statuses.value).for_each(
                            |(signature, maybe_status)| {
                                if let Some(status) = maybe_status {
                                    claim_transactions.remove(signature);

                                    if status.err.is_none() {
                                        num_landed_ok = num_landed_ok.saturating_add(1);
                                    } else {
                                        num_landed_error = num_landed_error.saturating_add(1);
                                    }
                                }
                            },
                        );
                    }
                    Err(e) => {
                        warn!("error reading signature statuses: {:?}", e);
                    }
                }
            }

            datapoint_info!(
                "claim_mev_tips-send_summary",
                ("claim_transactions_left", claim_transactions.len(), i64),
                ("num_send_ok", num_send_ok, i64),
                ("num_send_error", num_send_error, i64),
                ("num_landed_ok", num_landed_ok, i64),
                ("num_landed_error", num_landed_error, i64),
            );

            // give network a little extra time to process any lingering transactions
            // TODO (LB): need to think about the blockhash expiration thing above
            sleep(Duration::from_secs(5)).await;

            // resign with new blockhash and replace everything
            let blockhash = rpc_client.get_latest_blockhash().await?;
            claim_transactions = claim_transactions
                .into_iter()
                .map(|(signature, mut tx)| {
                    tx.sign(&[&keypair], blockhash);
                    (*tx.get_signature(), tx)
                })
                .collect();
        }
    }

    return Err(ClaimMevError::MaxSendTransactionRetriesExceeded {
        attempts: 0,
        remaining_transaction_count: 0, // TODO (LB)
        failed_transaction_count: 0,    // TODO (LB)
    });
}

fn build_transactions_new(
    tip_distribution_program_id: Pubkey,
    merkle_trees: &GeneratedMerkleTreeCollection,
    tdas: HashMap<Pubkey, Option<Account>>,
    claimants: HashMap<Pubkey, Option<Account>>,
    claim_status: HashMap<Pubkey, Option<Account>>,
    micro_lamports_per_compute_unit: u64,
    payer_pubkey: Pubkey,
) -> Vec<Transaction> {
    let tip_distribution_accounts: HashMap<Pubkey, TipDistributionAccount> = tdas
        .iter()
        .filter_map(|(pubkey, account)| {
            let account = account.as_ref()?;
            Some((
                *pubkey,
                TipDistributionAccount::try_deserialize(&mut account.data.as_slice()).ok()?,
            ))
        })
        .collect();

    let claim_statuses: HashMap<Pubkey, ClaimStatus> = claim_status
        .iter()
        .filter_map(|(pubkey, account)| {
            let account = account.as_ref()?;
            Some((
                *pubkey,
                ClaimStatus::try_deserialize(&mut account.data.as_slice()).ok()?,
            ))
        })
        .collect();

    datapoint_info!(
        "build_transactions_new",
        (
            "tip_distribution_accounts",
            tip_distribution_accounts.len(),
            i64
        ),
        ("claim_statuses", claim_statuses.len(), i64),
    );

    let tip_distribution_config =
        Pubkey::find_program_address(&[Config::SEED], &tip_distribution_program_id).0;

    let mut instructions = Vec::with_capacity(claimants.len());
    for tree in &merkle_trees.generated_merkle_trees {
        if tree.max_total_claim == 0 {
            continue;
        }

        // if unwrap panics, there's a bug in the merkle tree code because the merkle tree code relies on the state
        // of the chain to claim.
        let tip_distribution_account = tip_distribution_accounts
            .get(&tree.tip_distribution_account)
            .unwrap();

        // can continue here, as there might be tip distribution accounts this account doesn't upload for
        if tip_distribution_account.merkle_root.is_none() {
            continue;
        }

        for node in &tree.tree_nodes {
            // doesn't make sense to claim for claimants that don't exist anymore
            // can't claim for something already claimed
            // don't need to claim for claimants that get 0 MEV
            if claimants.get(&node.claimant).is_none()
                || claim_statuses.get(&node.claim_status_pubkey).is_some()
                || node.amount == 0
            {
                continue;
            }

            instructions.push(Instruction {
                program_id: tip_distribution_program_id,
                data: jito_tip_distribution::instruction::Claim {
                    proof: node.proof.clone().unwrap(),
                    amount: node.amount,
                    bump: node.claim_status_bump,
                }
                .data(),
                accounts: jito_tip_distribution::accounts::Claim {
                    config: tip_distribution_config,
                    tip_distribution_account: tree.tip_distribution_account,
                    claimant: node.claimant,
                    claim_status: node.claim_status_pubkey,
                    payer: payer_pubkey,
                    system_program: system_program::id(),
                }
                .to_account_metas(None),
            });
        }
    }

    // TODO (LB): see if we can do >1 claim here
    let transactions: Vec<Transaction> = instructions
        .into_iter()
        .map(|claim_ix| {
            let priority_fee_ix =
                ComputeBudgetInstruction::set_compute_unit_price(micro_lamports_per_compute_unit);
            Transaction::new_with_payer(&[priority_fee_ix, claim_ix], Some(&payer_pubkey))
        })
        .collect();

    transactions
}

/// heuristic to make sure we have enough funds to cover the rent costs if epoch has many validators
/// If insufficient funds, returns start balance, desired balance, and amount of sol to deposit
async fn is_sufficient_balance(
    payer: &Pubkey,
    rpc_client: &RpcClient,
    instruction_count: u64,
) -> Option<(u64, u64, u64)> {
    let start_balance = rpc_client
        .get_balance(payer)
        .await
        .expect("Failed to get starting balance");
    // most amounts are for 0 lamports. had 1736 non-zero claims out of 164742
    let min_rent_per_claim = rpc_client
        .get_minimum_balance_for_rent_exemption(ClaimStatus::SIZE)
        .await
        .expect("Failed to calculate min rent");
    let desired_balance = instruction_count
        .checked_mul(
            min_rent_per_claim
                .checked_add(DEFAULT_TARGET_LAMPORTS_PER_SIGNATURE)
                .unwrap(),
        )
        .unwrap();
    if start_balance < desired_balance {
        let sol_to_deposit = desired_balance
            .checked_sub(start_balance)
            .unwrap()
            .checked_add(LAMPORTS_PER_SOL)
            .unwrap()
            .checked_sub(1)
            .unwrap()
            .checked_div(LAMPORTS_PER_SOL)
            .unwrap(); // rounds up to nearest sol
        Some((start_balance, desired_balance, sol_to_deposit))
    } else {
        None
    }
}
