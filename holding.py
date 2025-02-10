#!/usr/bin/env python3

"""
FULL ASYNCHRONOUS CODE WITH NICEGUI IMPLEMENTATION
-------------------------------------------------
Converted from the provided Streamlit code into a NiceGUI application.
All I/O-bound functions have been converted to asynchronous using async/await.
File I/O operations utilize `aiofiles` for non-blocking behavior.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import os

import logging
import requests
import random
import re
import os
import json
from decimal import Decimal
from time import strftime, localtime
import pandas as pd
import plotly.express as px
import io
from nicegui import ui
from nicegui import app
import asyncio
import httpx
import aiofiles
from plotly.io import to_html
import aiohttp
from fastapi import Request, HTTPException
from solders.transaction import VersionedTransaction
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


from solders.keypair import Keypair
from solders.commitment_config import CommitmentLevel
from solders.rpc.requests import SendVersionedTransaction
from solders.rpc.config import RpcSendTransactionConfig
from surveillance import update_tracked_table,display_dashboard, update_non_moving_table,AsyncSessionLocal, get_user_key, store_user_key, setup_database
################# GLOBAL STYLING ###################
ui.add_body_html('''<script>
  function loadPage(page) {
    // This example uses fetch() to request new HTML from your endpoint.
    fetch(`/tracked-tokens?page=${page}`)
      .then(response => response.text())
      .then(html => {
        // Update the container element with the new HTML.
        document.getElementById("tracked-tokens-container").innerHTML = html;
      })
      .catch(error => console.error("Error loading page:", error));
  }
</script>
''')

# Inside holding.py

################# LOGGING CONFIG ###################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

################# SHYFT & ON-CHAIN CONFIG ##############
SHYFT_API_KEY = "z50w2Lk4A2ggBYYX"
SHYFT_BASE_URL = "https://api.shyft.to/sol/v1/transaction/history"
SOLANA_RPC_ENDPOINT = "https://rpc.shyft.to?api_key=z50w2Lk4A2ggBYYX"

######################## GLOBAL STATE ########################
from shared_state import SharedState
shared_state = SharedState()
logger = logging.getLogger(__name__)


################# FILES FOR JSON STORAGE ##################
MONITORED_ADDRESSES_FILE = "monitored_addresses.json"
ELIGIBLE_TOKENS_FILE = "eligible_tokens.json"
ALL_MONITORED_TOKENS_FILE = "all_monitored_tokens.json"
KOH_HISTORY_FILE = "king_of_the_hill_history.json"

######################## HELPER: LOAD/SAVE SESSION DATA ########################
##def load_session_data_key(key, default_value):
  #  """Emulates st.session_state.get(key, default)."""
   # return session_data.get(key, default_value)
######################no longer need as we don't st.session from streamlit 
#def set_session_data_key(key, value):
 #   """Emulates st.session_state[key] = value."""
  #  session_data[key] = value

######################## HELPER FUNCTIONS FOR FILE I/O ########################
async def load_monitored_addresses():
    """
    Asynchronously loads monitored addresses from file.
    """
    if os.path.exists(MONITORED_ADDRESSES_FILE):
        try:
            async with aiofiles.open(MONITORED_ADDRESSES_FILE, "r") as f:
                content = await f.read()
                addresses = json.loads(content)
                shared_state.set("monitored_tokens", addresses)
                return addresses
        except Exception as e:
            logger.error(f"Failed to load {MONITORED_ADDRESSES_FILE}: {str(e)}")
            return []
    else:
        await save_monitored_addresses([])
        return []

async def save_monitored_addresses(addresses):
    """
    Asynchronously saves monitored addresses to file.
    """
    try:
        async with aiofiles.open(MONITORED_ADDRESSES_FILE, "w") as f:
            await f.write(json.dumps(addresses, indent=4))
        shared_state.set("monitored_tokens", addresses)
    except Exception as e:
        logger.error(f"Failed to save {MONITORED_ADDRESSES_FILE}: {str(e)}")

async def load_eligible_tokens():
    """
    Asynchronously loads eligible tokens from file.
    """
    if os.path.exists(ELIGIBLE_TOKENS_FILE):
        try:
            async with aiofiles.open(ELIGIBLE_TOKENS_FILE, "r") as f:
                content = await f.read()
                tokens = json.loads(content)
                shared_state.set("eligible_tokens", tokens)
                return tokens
        except Exception as e:
            logger.error(f"Failed to load {ELIGIBLE_TOKENS_FILE}: {str(e)}")
            return []
    else:
        await save_eligible_tokens([])
        return []

async def save_eligible_tokens(tokens):
    """
    Asynchronously saves eligible tokens to file.
    """
    try:
        async with aiofiles.open(ELIGIBLE_TOKENS_FILE, "w") as f:
            await f.write(json.dumps(tokens, indent=4))
        shared_state.set("eligible_tokens", tokens)
    except Exception as e:
        logger.error(f"Failed to save {ELIGIBLE_TOKENS_FILE}: {str(e)}")

async def load_all_monitored_tokens():
    """
    Asynchronously loads all monitored tokens from file.
    """
    if os.path.exists(ALL_MONITORED_TOKENS_FILE):
        try:
            async with aiofiles.open(ALL_MONITORED_TOKENS_FILE, "r") as f:
                content = await f.read()
                tokens = json.loads(content)
                shared_state.set("all_monitored_tokens", tokens)
                return tokens
        except Exception as e:
            logger.error(f"Failed to load {ALL_MONITORED_TOKENS_FILE}: {str(e)}")
            return []
    else:
        await save_all_monitored_tokens([])
        return []

async def save_all_monitored_tokens(tokens):
    """
    Asynchronously saves all monitored tokens to file.
    """
    try:
        async with aiofiles.open(ALL_MONITORED_TOKENS_FILE, "w") as f:
            await f.write(json.dumps(tokens, indent=4))
        shared_state.set("all_monitored_tokens", tokens)
    except Exception as e:
        logger.error(f"Failed to save {ALL_MONITORED_TOKENS_FILE}: {str(e)}")

async def load_king_of_the_hill_history():
    """
    Asynchronously loads King of the Hill history from file.
    """
    if os.path.exists(KOH_HISTORY_FILE):
        try:
            async with aiofiles.open(KOH_HISTORY_FILE, "r") as f:
                content = await f.read()
                history = json.loads(content)
                shared_state.set("king_of_the_hill_history", history)
                return history
        except Exception as e:
            logger.error(f"Failed to load {KOH_HISTORY_FILE}: {str(e)}")
            return []
    else:
        await save_king_of_the_hill_history([])
        return []

async def save_king_of_the_hill_history(history):
    """
    Asynchronously saves King of the Hill history to file.
    """
    try:
        async with aiofiles.open(KOH_HISTORY_FILE, "w") as f:
            await f.write(json.dumps(history, indent=4))
        shared_state.set("king_of_the_hill_history", history)
    except Exception as e:
        logger.error(f"Failed to save {KOH_HISTORY_FILE}: {str(e)}")

######################## ON-CHAIN / SHYFT HELPERS ########################

async def async_send_rpc_request(payload, retries=5, backoff_factor=0.5, max_backoff=16):
    # Create spinner within a dedicated container if needed
    spinner_container = ui.column().classes('spinner-container')
    spinner = ui.spinner().classes('mr-2')
    

    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                response = await client.post(SOLANA_RPC_ENDPOINT, json=payload)
                if response.status_code == 200:
                    spinner.delete()         # Remove spinner on success
                    spinner_container.delete()  # Remove container if desired
                    return response.json()
                # ... handle retries ...
            except httpx.RequestError as e:
                # ... error handling ...
                pass
    spinner.delete()
    spinner_container.delete()
    logger.error("Max retries in async_send_rpc_request.")
    return {}


async def async_get_largest_token_holders_with_total_supply(token_mint, top_n=20):
    """
    Asynchronous version to get top N largest holders and mark the top one as bonding curve address.
    """
    pl = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [str(token_mint), {"commitment": "finalized"}]
    }
    res = await async_send_rpc_request(pl)
    if not res or "result" not in res:
        logger.error("Failed largest accounts => no data")
        return [], Decimal(0), None

    top_vals = res["result"]["value"][:top_n]
    t_acct_addrs = [v["address"] for v in top_vals]
    raw_amounts = [Decimal(v.get("amount", "0")) for v in top_vals]

    # The first holder is assumed to be the bonding curve address.
    bonding_curve_address = t_acct_addrs[0] if t_acct_addrs else None

    # fetch owners => getMultipleAccounts
    batch_p = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "getMultipleAccounts",
        "params": [t_acct_addrs, {"encoding": "jsonParsed", "commitment": "finalized"}]
    }
    batch_r = await async_send_rpc_request(batch_p)
    if not batch_r or "result" not in batch_r:
        owners = ["Unknown"] * len(t_acct_addrs)
    else:
        owners_info = batch_r["result"].get("value", [])
        owners = []
        for ai in owners_info:
            if not ai:
                owners.append("Unknown")
            else:
                parsed_d = ai.get("data", {}).get("parsed", {})
                inf = parsed_d.get("info", {})
                owners.append(inf.get("owner", "Unknown"))

    # total supply => getTokenSupply
    sup_p = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "getTokenSupply",
        "params": [str(token_mint), {"commitment": "finalized"}]
    }
    sup_r = await async_send_rpc_request(sup_p)
    if not sup_r or "result" not in sup_r or "value" not in sup_r["result"]:
        total_sup = Decimal(0)
        decimals = 0
    else:
        val = sup_r["result"]["value"]
        raw_sup = Decimal(val.get("amount", "0"))
        decimals = int(val.get("decimals", 0))
        total_sup = raw_sup / (Decimal(10) ** decimals) if decimals > 0 else raw_sup

    holders = []
    for own, raw_amt in zip(owners, raw_amounts):
        if total_sup > 0:
            adjusted = raw_amt / (Decimal(10) ** decimals)
            pct = (adjusted / total_sup) * 100
        else:
            adjusted = Decimal(0)
            pct = Decimal(0)
        holders.append((own, adjusted, pct))
    return holders, total_sup, bonding_curve_address

async def async_fetch_creator_wallet(mint_address):
    """
    Asynchronously fetch the creator wallet (developer wallet) for a given mint address.
    """
    if not mint_address or not is_valid_pubkey(mint_address):
        logger.error(f"Invalid mint address passed to async_fetch_creator_wallet: {mint_address}")
        return None

    url = f"https://frontend-api.pump.fun/coins/{mint_address}?sync=true"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={'accept': '*/*'}, timeout=10)
            response.raise_for_status()
            data = response.json()
            creator = data.get("creator", "Unknown")
            if creator == "Unknown":
                logger.warning(f"Creator not found for mint: {mint_address}")
            return creator
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch creator wallet for {mint_address}: {e}")
            return None

async def async_check_received_from_bonding_curve_via_shyft(holder_addr_str, bonding_curve_str, target_mint_str, tx_num=10):
    """
    Asynchronous check of inbound token transactions using httpx.
    """
    base_url = SHYFT_BASE_URL
    headers = {"x-api-key": SHYFT_API_KEY}
    params = {
        "network": "mainnet-beta",
        "tx_num": tx_num,
        "account": holder_addr_str,
        "enable_raw": "false"
    }

    total_recv = Decimal(0)
    results = []
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(base_url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            if data.get("success") and isinstance(data.get("result"), list):
                txs = data["result"]
                for tx_obj in txs:
                    sigs = tx_obj.get("signatures", ["NoSig"])
                    sig = sigs[0]
                    actions = tx_obj.get("actions", [])
                    for action in actions:
                        info = action.get("info", {})
                        token_addr = info.get("token_address", "")
                        amt_str = info.get("amount", "0")
                        source = info.get("source", "")
                        # Marking if the source is the bonding curve address
                        if token_addr == target_mint_str and source == bonding_curve_str:
                            results.append((sig, amt_str))
                            total_recv += Decimal(amt_str)
            else:
                logger.error("Shyft => inbound BC => unexpected data structure")
        except httpx.RequestError as e:
            logger.error(f"async_check_received_from_bonding_curve_via_shyft => {e}")
    return total_recv, results

async def async_check_outgoing_via_shyft(wallet_addr_str, target_mint_str, tx_num=10):
    """
    Asynchronous check of outbound token transactions using httpx.
    """
    base_url = SHYFT_BASE_URL
    headers = {"x-api-key": SHYFT_API_KEY}
    params = {
        "network": "mainnet-beta",
        "tx_num": tx_num,
        "account": wallet_addr_str,
        "enable_raw": "false"
    }

    total_out = Decimal(0)
    outgoing_data = []
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(base_url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            if data.get("success") and isinstance(data.get("result"), list):
                for tx_obj in data["result"]:
                    sigs = tx_obj.get("signatures", ["NoSig"])
                    sig = sigs[0]
                    for action in tx_obj.get("actions", []):
                        info = action.get("info", {})
                        token_addr = info.get("token_address", "")
                        amt_str = info.get("amount", "0")
                        source = info.get("source", "")
                        if token_addr == target_mint_str and source == wallet_addr_str:
                            outgoing_data.append((sig, amt_str))
                            total_out += Decimal(amt_str)
            else:
                logger.error("Shyft => dev wallet => unexpected data structure")
        except httpx.RequestError as e:
            logger.error(f"async_check_outgoing_via_shyft => {e}")
        except Exception as e:
            logger.error(f"Unknown error => async_check_outgoing_via_shyft => {e}")
    return total_out, outgoing_data

def calculate_velocity(history, current_index=0, previous_index=1):
    """
    Calculates the percentage increase/decrease in 'usd_market_cap' between
    two points in history.
    """
    try:
        current_mc = Decimal(history[current_index].get("usd_market_cap", 0))
        prev_mc = Decimal(history[previous_index].get("usd_market_cap", 0))
        if prev_mc == 0:
            return Decimal(0)
        return ((current_mc - prev_mc) / prev_mc) * 100
    except (IndexError, Decimal.InvalidOperation) as e:
        logger.error(f"Velocity calc => {e}")
        return Decimal(0)

################ MONITORING (Used in "Criteria Monitoring") ############

async def get_token_holdings_async(token_mint, wallet_addr):
    """
    Asynchronously returns how many tokens 'wallet_addr' holds of 'token_mint'.
    """
    pl = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [wallet_addr, {"mint": token_mint}, {"encoding": "jsonParsed"}]
    }
    resp = await async_send_rpc_request(pl)
    if not resp or "result" not in resp:
        return 0
    total_amt = Decimal(0)
    for acct in resp["result"]["value"]:
        info = acct.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
        bal = info.get("tokenAmount", {}).get("uiAmount", 0)
        total_amt += Decimal(bal)
    return float(total_amt)

async def get_whale_holdings_async(token_mint, top_n=20):
    """
    Asynchronously returns the % of total supply held by the largest holder.
    """
    holders, total_supply, bc_address = await async_get_largest_token_holders_with_total_supply(token_mint, top_n)

    if holders and total_supply > 0:
        return float(holders[0][2])  # holders[0][2] => % of total
    return 0.0

async def get_token_info_async(token_mint):
    """
    Asynchronously fetch token info from an external API.
    """
    url = f"https://frontend-api.pump.fun/coins/{token_mint}?sync=true"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={"accept": "*/*"}, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                "name": data.get("name", "Unknown"),
                "symbol": data.get("symbol", "UNKNOWN"),
                "total_supply": float(data.get("total_supply", 0)),
                "usd_market_cap": float(data.get("usd_market_cap", 0)),
                "creator": data.get("creator", "Unknown")
            }
        except httpx.RequestError as e:
            logger.error(f"Error fetching token info for {token_mint}: {e}")
            return {
                "name": "Unknown",
                "symbol": "UNKNOWN",
                "total_supply": 0.0,
                "usd_market_cap": 0.0,
                "creator": "Unknown"
            }

async def async_fetch_token_details(mint_address, dev_wallet):
    """
    Asynchronously fetch token details such as total supply using mint and dev wallet.
    """
    if not mint_address or not is_valid_pubkey(mint_address):
        logger.error(f"Invalid mint address passed to async_fetch_token_details: {mint_address}")
        return None

    if not dev_wallet or not is_valid_pubkey(dev_wallet):
        logger.error(f"Invalid dev wallet address: {dev_wallet}")
        return None

    url = f"https://frontend-api.pump.fun/coins/{mint_address}?sync=true"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={'accept': '*/*'}, timeout=10)
            response.raise_for_status()
            data = response.json()
            total_supply = float(data.get("total_supply", 0))
            return {
                "creator": dev_wallet,
                "total_supply": total_supply,
                "name": data.get("name", "Unknown"),
                "symbol": data.get("symbol", "UNKNOWN"),
            }
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch token details for {mint_address}: {e}")
            return None

async def monitor_transfers_async():
    """
    Asynchronously monitor transfers for addresses in session_data.
    Only runs if there are monitored addresses.
    """
    monitored = await load_monitored_addresses()
    if not monitored:
        logger.info("No monitored addresses. Skipping monitoring.")
        return

    logger.info(f"Monitoring {len(monitored)} addresses.")

    eligible = await load_eligible_tokens()
    koh_hist = await load_king_of_the_hill_history()

    for addr in monitored:
        txs = []  # Placeholder for async_get_recent_transactions(addr) since not defined above
        # Iterate through txs if implemented...
        # ...
        # (Existing logic using txs)

    await save_eligible_tokens(eligible)
    logger.info(f"Monitoring complete. Found {len(eligible)} eligible tokens.")

async def sync_tokens_to_koh_history_async():
    """
    Asynchronously sync newly discovered tokens into KOH_HISTORY_FILE so they appear in KoH.
    """
    tracked_tokens = shared_state.get("tracked_tokens") or {}
    if not tracked_tokens:
        logger.info("No tracked tokens in session data. Skipping sync_tokens_to_koh_history.")
        return

    current_history = await load_king_of_the_hill_history()
    existing_symbols = {h["symbol"] for h in current_history if "symbol" in h}
    existing_mints = {h["mint"] for h in current_history if "mint" in h}

    for symbol, detail in tracked_tokens.items():
        if symbol not in existing_symbols and detail.get("address") not in existing_mints:
            mint_address = detail.get("address", "")
            if not mint_address or not is_valid_pubkey(mint_address):
                logger.warning(f"Invalid mint address: {mint_address}. Skipping.")
                continue

            dev_wallet = await async_fetch_creator_wallet(mint_address)
            if not dev_wallet:
                logger.warning(f"No dev wallet found for mint: {mint_address}. Skipping.")
                continue

            token_details = await async_fetch_token_details(mint_address, dev_wallet)
            if token_details:
                new_record = {
                    "name": token_details.get("name", "Unknown"),
                    "symbol": token_details.get("symbol", symbol),
                    "mint": mint_address,
                    "creator": token_details.get("creator", "Unknown"),
                    "usd_market_cap": float(detail.get("market_cap", 0.0)),
                    "total_supply": float(token_details.get("total_supply", 0.0)),
                    "usd_price": float(detail.get("usd_price", 0.0)),
                    "timestamp": strftime("%Y-%m-%d %H:%M:%S", localtime())
                }
            else:
                logger.error(f"Failed to fetch token details for mint: {mint_address}")
                new_record = {
                    "name": symbol,
                    "symbol": symbol,
                    "mint": mint_address,
                    "creator": "Unknown",
                    "usd_market_cap": float(detail.get("market_cap", 0.0)),
                    "total_supply": 0.0,
                    "usd_price": 0.0,
                    "timestamp": strftime("%Y-%m-%d %H:%M:%S", localtime())
                }

            current_history.insert(0, new_record)
            existing_symbols.add(symbol)
            existing_mints.add(mint_address)

    await save_king_of_the_hill_history(current_history)
    logger.info("sync_tokens_to_koh_history completed successfully.")

################### NICEGUI APP STARTS HERE ############################

#async def init_session():
 #   """Initialize dictionary keys similarly to session_init()."""
  #  if "tracked_tokens" not in session_data:
   #     session_data["tracked_tokens"] = {}
    #if "graduated_tokens" not in session_data:
    #    session_data["graduated_tokens"] = []
 #   if "market_cap_history" not in session_data:  ########Same here we no longer use the session state
  #      session_data["market_cap_history"] = {}
   # if "velocities" not in session_data:
  #      session_data["velocities"] = {}
  #  if "selected_token" not in session_data:
   #     session_data["selected_token"] = None

# Initialize session data
# Initialize session data before starting the GUI event loop
#asyncio.run(init_session())

################################################
#  AUTO-REFRESH LOGIC VIA ui.timer()
#  We'll set up a 5-minute timer to run monitor_transfers_async()
################################################

import os

BASE_URL = "https://on-air.io/trinity/device-0/"

################################################
#  GLOBAL HEADER WITH DARK MODE TOGGLE & RESPONSIVE NAV
################################################
def setup_navigation():
    with ui.header(elevated=True).classes('items-center justify-between w-full'):
        ui.label("King Of The Hill").classes('ml-4 text-xl font-bold text-center')

        with ui.row():
            ui.button("Overview", on_click=lambda: ui.run_javascript(f"window.location.href='{BASE_URL}'"))
            ui.button("King of the Hill", on_click=lambda: ui.run_javascript(f"window.location.href='{BASE_URL}koh'"))

            ui.button("Token Analysis", on_click=lambda: ui.run_javascript(f"window.location.href='{BASE_URL}analysis'"))
            ui.button("Pump Labs", on_click=lambda: ui.run_javascript(f"window.location.href='{BASE_URL}pump-labs'"))
            ui.button("Pump-Bot", on_click=lambda: ui.run_javascript(f"window.location.href='{BASE_URL}trade'"))
# Call this function at app startup
setup_navigation()



logger = logging.getLogger(__name__)
shared_state = SharedState()


###############################################################################
# 1) SET UP A GLOBAL STARTUP HOOK FOR BACKGROUND TASKS
###############################################################################
@app.on_startup
async def start_background_tasks():
    
    """
    Schedules your infinite surveillance logic in the background
    so it keeps running regardless of which page the user is on.
    """
    logger.info("=== Starting background tasks at startup ===")

    # Example: launch the infinite `run_surveillance_dashboard_async()` loop.
    # It might do all your token monitoring, data updates, etc.
#    asyncio.create_task(run_surveillance_dashboard_async())
#    asyncio.create_task(display_dashboard)
    # If you have a periodic refresh function, you can also schedule it
    # for every 5 minutes (300 seconds), for example:
    #from surveillance import refresh_dashboard_data_async
    #ui.timer(
    #    3,
    #    lambda: asyncio.create_task(refresh_dashboard_data_async()),
    #)
    
    koh_history = shared_state.get("koh_history") or []
    if not koh_history:
        try:
            shared_state.set("koh_history", await load_king_of_the_hill_history())
            logger.info("KoH history loaded successfully.")
        except Exception as e:
            # handle error
            koh_history = shared_state.get("koh_history") or []

        return

    # Retrieve koh_history from global_state
    koh_history = shared_state.get("koh_history") or []
    
    # Create a container for the page content
    container = ui.column().classes('m-4 p-4 shadow-md rounded-lg')
    try:
        if koh_history:
            opt_labels = [f"{h['name']} ({h['symbol']})" for h in koh_history]
            
            # Retrieve the previously selected token from the global state
            selected_token = shared_state.get("selected_token")

            pre_selected_label = (
                f"{selected_token['name']} ({selected_token['symbol']})"
                if selected_token and selected_token in koh_history
                else None
            )

            # Handle token selection
            async def on_koh_select_change(e):
                try:
                    selected_index = opt_labels.index(e.value)
                    shared_state.set("selected_token", koh_history[selected_index])
                    ui.notify(f"Selected token: {e.value}", type='info')
                except ValueError:
                    ui.notify("Invalid token selected.", type='negative')

            # Create dropdown for token selection with pre-selected token
            ui.select(
                opt_labels,
                value=pre_selected_label,
                on_change=on_koh_select_change,
                label="Pick a KoH token for analysis:"
            ).classes('mt-4')
        else:
            ui.notify(
                "No KoH history yet. Possibly run King of the Hill or the Surveillance dashboard?",
                type='info'
            )
    except Exception as e:
        ui.notify(f"Failed to load KoH history: {e}", type='negative')
        logger.error(f"KoH history loading error: {e}")

    container = ui.column().classes('m-4 p-4 shadow-md rounded-lg')
    # Attempt to restore the tracked table from storage if available
    #stored_table_html = app.storage.general.get('tracked_table_html', None)
    #if stored_table_html:
    #    # Assume tracked_table_container is a globally accessible container for the table
    #    with tracked_table_container:
    #        tracked_table_container.clear()
    #        ui.html("<h2>Currently Tracked Tokens</h2>")
    #        ui.html(stored_table_html)
    #else:
        # If no stored table, update it freshly
     #   await update_tracked_table()
############################################################################
# PAGE 1: OVERVIEW  (Originally the "Overview" tab)
############################################################################
@ui.page('/')

def main():
    
    setup_navigation()
    # -- Add the script we want for THIS page’s background --
    ui.add_head_html('''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/tengbao/vanta/dist/vanta.dots.min.js"></script>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        VANTA.DOTS({
          el: "#vanta-bg",
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          backgroundColor: 0x222222,
          color: 0xff8820,
          color2: 0xff8820,
          size: 3.00,
          spacing: 35.00,
          showLines: true
        });
    });
    </script>
    ''')

    # -- Add the style for the background container --
    ui.add_head_html('''
    <style>
      #vanta-bg {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: -1;
      }
    </style>
    ''')

    # -- Finally, place the background DIV for this page --
    ui.html('<div id="vanta-bg"></div>')
   
    ui.label("Welcome to the Full Holding Dashboard!").classes('text-xl font-bold')
    ui.markdown("Use the buttons above to navigate to other sections.")

    with ui.card().classes('m-4 p-4 shadow-md rounded-lg'):
        ui.label("Welcome to the Full Holding Dashboard!").classes('text-xl font-bold')
        ui.markdown('''
Welcome to the Full Holding Dashboard!

**Full Holding Dashboard** is designed to provide comprehensive insights into your monitored Solana wallets and tokens. Use the navigation links above to explore:

- [**King of the Hill**](/koh): Access the Surveillance Dashboard to monitor top tokens.
- [**Token Analysis**](/analysis): Dive deep into specific token analytics.
- [**Pump Labs**](/pump-labs): Experiment with our Pump Labs.
- [**Pump-Bot**](/trade): Access the Pump-Bot trading interface.
''')

############################################################################
# PAGE 2: KING OF THE HILL  (Originally koh_tab)
############################################################################
from nicegui import ui, app
import asyncio
import time
async def initialize_page():
    if not (shared_state.get("page_initialized") or False):
        shared_state.set("content_data", "Loaded Content")
        shared_state.set("page_initialized", True)

############################### Ccopied the enteir update-tracked table becasue session state is making it hard to access the table without issue###############################
def trigger_file_change():
    """Touch a monitored file to trigger a NiceGUI reload."""
    with open('holding.py', 'a') as f:
        f.write(f'\n# Triggered reload at {time.time()}')#### ############################### ###############################


filter_inputs = {}
@ui.page('/koh')
async def page_koh():
    
    ui.add_head_html('''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/tengbao/vanta/dist/vanta.waves.min.js"></script>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        VANTA.WAVES({
          el: "#vanta-bg",
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0x38596b,
          shininess: 50.00,
          waveHeight: 20.00,
          waveSpeed: 1.00,
          zoom: 1.05
        });
    });
    </script>
    ''')

    ui.add_head_html('''
    <style>
      #vanta-bg {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: -1;
      }
    </style>
    ''')

    ui.html('<div id="vanta-bg"></div>')
  
    
     # Somewhere in your "KOH" page layout:
    
        
        

    
    setup_navigation()
    
    

    
    # set up a container for your table
    with ui.column().classes("w-1/4") as non_moving_container:
        ui.label("Stagnant Tokens (5m Δ0%)").classes("text-lg font-bold")
    with ui.column() as table_container:
        ui.label("Current Dashboard")

    with ui.column() as filter_ui_container:
        ui.label("Apply Filter Criteria").classes("text-lg font-bold")

    with ui.column() as filtered_table_container:
        ui.label("Filtered Tokens Result Below:").classes("text-md font-bold")

    
    
    

    # 1) on first page load, build & show table
    
    await update_tracked_table(table_container)
    
    await update_non_moving_table(non_moving_container)
    
    # 2) define a timer-based refresh function
    async def refresh_local_table():
        try:
            await update_tracked_table(table_container)
            
            
            
            logger.info("Local table refreshed successfully.")
        except Exception as e:
            ui.notify(f"Failed to refresh local table: {e}", type='negative')
            logger.exception("Local refresh error")

    async def refresh_local_table1():
        try:
            
            await update_non_moving_table(non_moving_container)
            logger.info("Local table refreshed successfully.")
        except Exception as e:
            ui.notify(f"Failed to refresh local table: {e}", type='negative')
            logger.exception("Local refresh error")

    # refresh every 3s
    ui.timer(3, lambda: asyncio.create_task(refresh_local_table()))
    ui.timer(300, lambda: asyncio.create_task(refresh_local_table1()))
    ui.timer(300, lambda: ui.notify('Non-moving table data fetched'))
    
    # possibly do other UI items or finalize
    ui.update()
    #    from surveillance import update_tracked_table,run_surveillance_dashboard_async, refresh_dashboard_data_wrapper, refresh_dashboard_data_async,display_dashboard, tracked_table_container, build_chart_section, update_data, setup_database, init_telegram_session
        
        
    #    await run_surveillance_dashboard_async()
    #    await display_dashboard()
        
       
    
        
        
        #loading_label.text = "Dashboard loaded successfully!"
    #except Exception as e:
    #    ui.notify(f"Failed to display the dashboard: {e}", type='negative')
     #   logger.error(f"Dashboard display error: {e}")
     #   return

    # Ensure koh_history is loaded in global_state
    #koh_history = shared_state.get("koh_history") or []
    #if not koh_history:
    #    try:
    #        shared_state.set("koh_history", await load_king_of_the_hill_history())
     #       logger.info("KoH history loaded successfully.")
      #  except Exception as e:
       #     # handle error
            #koh_history = shared_state.get("koh_history") or []

        #return

    # Retrieve koh_history from global_state
    #koh_history = shared_state.get("koh_history") or []
    
    # Create a container for the page content
    
    # Note: Removed container.clear() since the container is new and empty by default

    #with container:
    #    ui.label("👑 King of the Hill => Surveillance Dashboard").classes('text-xl font-bold')
    #    loading_label = ui.label("Loading Token Surveillance Dashboard... (please wait)")

    # Try to load the display dashboard
    
   
    # Build token selection dropdown
    

    

   # async def refresh_data():
   #     try:
   #         await refresh_dashboard_data_async()
   #         logger.info("Dashboard data refreshed successfully.")
   #     except Exception as e:
    #        logger.error(f"Failed to refresh dashboard data: {e}")
#
    # Save the refresh task in global_state
 #   shared_state.set("refresh_task", asyncio.create_task(refresh_data()))

    # Periodically refresh dashboard data
    #ui.timer(3, lambda: asyncio.create_task(refresh_dashboard_data_wrapper()))
    #ui.timer(3, lambda: asyncio.create_task(update_data()))





################################################
#  PAGE 3: TOKEN ANALYSIS  (Originally analysis_tab)
################################################

shared_components = {}

KOH_TOKENS_JSON = "king_of_the_hill_tokens.json"

async def load_koh_tokens():

    """Asynchronously load tracked tokens from the JSON file."""
    if os.path.exists(KOH_TOKENS_JSON):
        try:
            async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
                content = await f.read()
                tokens = json.loads(content)
                return tokens
        except Exception as e:
            logger.error(f"Failed to load tokens from {KOH_TOKENS_JSON}: {e}")
    return []


#---------------------- API ENDPOINTS FOR BUY/SELL ---------------
# ----------------------- API ENDPOINTS FOR BUY/SELL ---------------
# Define the signature validation function
def is_valid_signature(signature):
    """
    Validates that the signature is a 88-character Base58 string.
    """
    return bool(re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{88}', signature))

################### API ENDPOINTS FOR BUY/SELL ############

@app.post("/connect_balance")
async def connect_balance(request: Request):
    """
    Expects JSON:
      { "user_id": 1 }
    Looks up that user's private key in DB,
    decodes + fetches SOL balance from mainnet, returns { public_key, balance_sol } or error.
    """
    data = await request.json()
    user_id = int(data["user_id"])

    # 1) Get private key from DB
    async with AsyncSessionLocal() as session:
        row = await get_user_key(user_id, session)
        if not row:
            return {"error": "No private key found in DB for this user_id."}

    private_key_b58 = row.private_key
    try:
        kp = Keypair.from_base58_string(private_key_b58)
    except Exception as e:
        return {"error": f"Invalid private key in DB: {e}"}

    # 2) Query SOL balance
    import httpx
    from solders.rpc.requests import GetBalance
    rpc_url = "https://api.mainnet-beta.solana.com"
    payload = GetBalance(kp.pubkey()).to_json()

    async with httpx.AsyncClient() as client:
        resp = await client.post(rpc_url, json=payload)
        if resp.status_code != 200:
            return {"error": f"Balance fetch error: status={resp.status_code}"}
        rj = resp.json()
        lamports = rj.get("result", {}).get("value", 0)
        sol_balance = lamports / 1e9

    return {
        "public_key": str(kp.pubkey()),
        "balance_sol": sol_balance
    }
app = FastAPI()
@app.get("/tracked-tokens", response_class=HTMLResponse)
async def tracked_tokens(page: int = Query(1), items_per_page: int = Query(10)):
    html = await build_tracked_table_html(page, items_per_page)
    return html
@app.post("/api/trade")
async def trade_endpoint(request: Request):
    """
    Expects JSON body:
      {
        "user_id": <int>,
        "symbol": "ABC",
        "action": "buy" or "sell",
        "amount": e.g. "0.1" or "100%"
      }

    1) Look up user private key from DB
    2) Look up token's mint from the king_of_the_hill_tokens.json
    3) Create transaction with PumpPortal
    4) Sign + send to Solana
    5) Return txSignature or error
    """
    body = await request.json()
    user_id = int(body["user_id"])
    symbol = body["symbol"]
    action = body["action"]
    amount = body["amount"]
    slippage = float(body.get("slippage", 10))
    priority_fee = float(body.get("priorityFee", 0.005))
    # Before signing the transaction
    
    # 1) Fetch private key from DB
    async with AsyncSessionLocal() as session:
        row = await get_user_key(user_id, session)
        if not row:
            logger.error(f"No user key found for user_id={user_id}")
            return {"error": f"No user key found for user_id={user_id}"}
    private_key_b58 = row.private_key

   

    # 2) Reconstruct keypair
    try:
        kp = Keypair.from_base58_string(private_key_b58)
    except Exception as e:
        logger.error(f"Invalid private key: {e}")
        return {"error": f"Invalid private key: {e}"}

    # 3) Load KOH tokens from JSON, find the token with that symbol
    tokens_list = await load_koh_tokens()
    token_match = next((t for t in tokens_list if t.get("symbol") == symbol), None)
    if not token_match:
        logger.error(f"No token found in KOH JSON for symbol '{symbol}'.")
        return {"error": f"No token found in KOH JSON for symbol '{symbol}'."}

    mint_address = token_match["address"]  # e.g., "hoJnC5VTJomZoSfPXHvHLzzVUW4PEqXC2QSiuNkHump"

    # 4) Build request to PumpPortal
    trade_local_url = "https://pumpportal.fun/api/trade-local"
    denominated_in_sol = "true" if action == "buy" else "false"
    req_body = {
        "publicKey": str(kp.pubkey()),
        "action": action,
        "mint": mint_address,
        "amount": amount,
        "denominatedInSol": denominated_in_sol,
        "slippage": slippage,
        "priorityFee": priority_fee,
        "pool": "auto"
    }

    async with httpx.AsyncClient() as client:
        try:
            # Use form-encoded data as per original implementation
            resp = await client.post(trade_local_url, data=req_body)
            resp.raise_for_status()
            logger.info(f"PumpPortal response status: {resp.status_code}")
        except httpx.HTTPStatusError as e:
            logger.error(f"PumpPortal HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": f"PumpPortal HTTP error: {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.error(f"PumpPortal request error: {e}")
            return {"error": "PumpPortal request failed."}

        # 5) Deserialize & sign
        try:
            serialized_tx_bytes = resp.content
            tx = VersionedTransaction.from_bytes(serialized_tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [kp])
            logger.info("Transaction deserialized and signed successfully.")
        except Exception as e:
            logger.error(f"Transaction deserialization/signing error: {e}")
            return {"error": f"Transaction error: {e}"}
        payload = SendVersionedTransaction(
            signed_tx,
            RpcSendTransactionConfig(preflight_commitment=CommitmentLevel.Confirmed)
        ).to_json()
        # 6) Send to Solana RPC with retries and proper confirmation
        rpc_url = "https://api.mainnet-beta.solana.com"
        try:
            rpc_response = await client.post(rpc_url, headers={"Content-Type": "application/json"}, data=payload)
            rpc_response.raise_for_status()
            logger.info(f"Solana RPC response status: {rpc_response.status_code}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Solana RPC HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": f"Solana RPC HTTP error: {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.error(f"Solana RPC request error: {e}")
            return {"error": "Solana RPC request failed."}

        result_json = rpc_response.json()
        tx_signature = result_json.get("result")
        if not tx_signature:
            logger.error(f"RPC missing result: {rpc_response.text}")
            return {"error": "RPC missing transaction signature."}

        logger.info(f"Transaction successful. Signature: {tx_signature}")
        return {"txSignature": tx_signature}
@ui.page('/analysis')
async def page_token_analysis():
    ui.add_head_html('''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/tengbao/vanta/dist/vanta.net.min.js"></script>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        VANTA.NET({
          el: "#vanta-bg",
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0x6b515a,
          backgroundColor: 0x23153c,
          points: 10,
          maxDistance: 20,
          spacing: 15,
          showDots: true
        });
    });
    </script>
    ''')

    ui.add_head_html('''
    <style>
      #vanta-bg {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: -1;
      }
    </style>
    ''')

    ui.html('<div id="vanta-bg"></div>')
    
    
   
    setup_navigation()
    with ui.card().classes('m-4 p-4 shadow-md rounded-lg'):
        ui.label("Token Analysis (Steps 0–4)").classes('text-xl font-bold')

        tracked_tokens_list = await load_koh_tokens()

        tracked_tokens = {token["symbol"]: token for token in tracked_tokens_list}
        if not tracked_tokens:
            ui.notify("No tracked tokens available. Please ensure tokens are being tracked in the 'King of the Hill' page.", type='info')
            return

        if isinstance(tracked_tokens_list, dict):
            tracked_tokens_list = list(tracked_tokens_list.values())

        for token in tracked_tokens_list:
            if "mint" not in token:
                token["mint"] = token.get("address", "")
            if "creator" not in token:
                token["creator"] = token.get("creator_wallet", "")

        token_symbols = list(tracked_tokens.keys())
        if not token_symbols:
            ui.notify("No valid tokens found for analysis.", type='error')
            return

        selected_symbol = ui.select(
            label="Select a Token for Analysis",
            options=token_symbols,
            value=token_symbols[0],
            on_change=lambda e: asyncio.create_task(update_selected_token(e.value))
        ).classes('w-1/4')

        sel_token = tracked_tokens[token_symbols[0]]

        async def update_selected_token(symbol):
            nonlocal sel_token
            sel_token = tracked_tokens.get(symbol, {})
            if not sel_token:
                ui.notify(f"Token '{symbol}' not found in tracked tokens.", type='warning')
                return

            token_info_container.clear()
            analysis_steps_container.clear()

            with token_info_container:
                ui.markdown(f"### Analyzing: {sel_token.get('name','N/A')} ({sel_token.get('symbol','N/A')})")
                ui.markdown(f"**Mint Address:** `{sel_token.get('mint','N/A')}`")
                ui.markdown(f"**Dev Wallet (Creator):** `{sel_token.get('creator','N/A')}`")
                ui.markdown(f"**Timestamp:** `{sel_token.get('timestamp','N/A')}`")

            await build_analysis_steps()

        token_info_container = ui.column().classes('w-full items-start mt-4')
        with token_info_container:
            ui.markdown(f"### Analyzing: {sel_token.get('name','N/A')} ({sel_token.get('symbol','N/A')})")
            ui.markdown(f"**Mint Address:** `{sel_token.get('mint','N/A')}`")
            ui.markdown(f"**Dev Wallet (Creator):** `{sel_token.get('creator','N/A')}`")
            ui.markdown(f"**Timestamp:** `{sel_token.get('timestamp','N/A')}`")

        analysis_steps_container = ui.column().classes('w-full items-start mt-4')

        async def build_analysis_steps():
            with analysis_steps_container:
                ui.markdown("#### Step 0: On-chain RPC Check")
                test_pl = {"jsonrpc": "2.0", "id": 1, "method": "getVersion", "params": []}
                test_resp = await async_send_rpc_request(test_pl)
                if test_resp and "result" in test_resp:
                    ui.notify("✅ On-chain RPC connected.", type='positive')
                else:
                    ui.notify("❌ Could not connect on-chain. Stopping analysis steps.", type='negative')
                    return

                ui.markdown("#### Step 1: Largest Holders")
                with ui.row().classes('w-full justify-start items-center'):
                    ui.label("Number of Top Holders").classes('text-lg font-bold mt-4')
                    holder_slider = ui.slider(min=5, max=50, value=20, step=1).classes('ml-4')

                largest_holders_container = ui.column().classes('w-full items-center mt-4')

                async def load_largest_holders():
                    largest_holders_container.clear()
                    top_n = int(holder_slider.value)
                    holders, total_supply, bonding_curve_address = await async_get_largest_token_holders_with_total_supply(
                        sel_token["mint"], top_n
                    )
                    if not holders:
                        ui.notify("No largest holders found. Possibly invalid mint address.", type='warning')
                        return

                    with ui.column():
                        ui.markdown(f"**Total Supply:** {float(total_supply):,.4f} tokens").classes('text-md mt-4')

                    df = pd.DataFrame(holders, columns=["Holder Address", "Amount", "% of Total"])
                    df["Note"] = ""
                    if not df.empty:
                        df.at[0, "Note"] = "🎯 Bonding Curve"

                    table_data = df.to_dict(orient='records')

                    with ui.column():
                        ui.label("Largest Token Holders Table").classes('text-md font-bold mb-2')
                        ui.table(
                            columns=[
                                {'name': 'Holder Address', 'label': 'Holder Address', 'field': 'Holder Address', 'sortable': True},
                                {'name': 'Amount', 'label': 'Amount', 'field': 'Amount', 'sortable': True},
                                {'name': '% of Total', 'label': '% of Total', 'field': '% of Total', 'sortable': True},
                                {'name': 'Note', 'label': 'Note', 'field': 'Note'},
                            ],
                            rows=table_data
                        ).classes('w-3/4 mt-4').props('rows-per-page-options="[10, 20, 50]"')

                    owners = [h[0] for h in holders]
                    shared_components['inbound_holder_select'] = ui.select(
                        label="Choose a holder to see inbound from BC",
                        options=owners
                    ).classes('w-1/2 mt-2')
                    ui.notify("Holder dropdown populated.", type='info')

                    top10 = df.head(10).copy()
                    sum_others = df["Amount"].sum() - top10["Amount"].sum()
                    if sum_others > 0:
                        new_row = {
                            "Holder Address": "Others",
                            "Amount": sum_others,
                            "% of Total": (sum_others / total_supply) * 100 if total_supply > 0 else 0,
                            "Note": ""
                        }
                        top10 = pd.concat([top10, pd.DataFrame([new_row])], ignore_index=True)

                    fig_pie = px.pie(
                        top10,
                        names="Holder Address",
                        values="Amount",
                        title="Largest Token Holders Distribution (Top 10 + Others)"
                    )

                    fig_html = to_html(fig_pie, full_html=False)
                    with ui.column():
                        ui.label("Token Holders Distribution (Pie Chart)").classes('text-md font-bold mb-2')
                        ui.add_body_html(fig_html)

                ui.button("Load Largest Holders", on_click=load_largest_holders).classes('mt-4')

                ui.markdown("#### Step 2: Choose Number of Signatures to Fetch 📝")
                user_sig_count = ui.input(label="Signatures to fetch per holder", value="10").classes('w-1/4 mt-2')

                async def show_inbound_from_bc():
                    chosen_holder = shared_components.get('inbound_holder_select').value
                    if not chosen_holder:
                        ui.notify("No holder selected", type='warning')
                        return

                    try:
                        tx_num = int(user_sig_count.value)
                    except ValueError:
                        ui.notify("Invalid signature count. Must be an integer.", type='negative')
                        return

                    bonding_curve = shared_state.get("bc_address") or ""


                    total_recv, inbound_sigs = await async_check_received_from_bonding_curve_via_shyft(
                        holder_addr_str=chosen_holder,
                        bonding_curve_str=bonding_curve,
                        target_mint_str=sel_token["mint"],
                        tx_num=tx_num
                    )
                    ui.markdown(f"**Total Received** => {float(total_recv):,.4f}, # TX => {len(inbound_sigs)}")
                    if inbound_sigs:
                        sig_options = [f"{sig} => {amt_str}" for (sig, amt_str) in inbound_sigs]
                        sig_select = ui.select(sig_options, label="Pick a transaction signature to see details").classes('w-1/2 mt-2')

                        async def on_sig_pick(e):
                            ui.notify(f"You chose => {e.value}", type='info')

                        sig_select.on('update', on_sig_pick)
                    else:
                        ui.notify("No inbound TX from bonding curve found.", type='info')

                ui.button("Check Inbound from BC", on_click=show_inbound_from_bc).classes('mt-2')

                ui.markdown("#### Step 3: Dev Wallet Outgoing 🚀")
                dev_addr = sel_token.get("creator", "N/A")
                ui.markdown(f"Dev wallet => `{dev_addr}` (signature count => from input above)").classes('mt-2')

                async def check_dev_wallet_outgoing():
                    try:
                        tx_num = int(user_sig_count.value)
                    except ValueError:
                        ui.notify("Invalid signature count. Must be an integer.", type='negative')
                        return
                    dev_total_out, dev_data = await async_check_outgoing_via_shyft(dev_addr, sel_token["mint"], tx_num)

                    ui.markdown(f"**Dev Wallet Outgoing** => {float(dev_total_out):,.4f} tokens, # TX => {len(dev_data)}").classes('mt-2')
                    if dev_data:
                        out_df = pd.DataFrame(dev_data, columns=["Signature", "Amount"])
                        out_records = out_df.to_dict(orient='records')
                        ui.table(
                            columns=[
                                {'name': 'Signature', 'label': 'Signature', 'field': 'Signature'},
                                {'name': 'Amount', 'label': 'Amount', 'field': 'Amount'},
                            ],
                            rows=out_records,
                        ).classes('w-1/2 mt-2')
                    else:
                        ui.notify("No dev wallet outgoing found for that signature count & mint.", type='info')

                ui.button("Check Dev Wallet Outgoing", on_click=check_dev_wallet_outgoing).classes('mt-2')

                async def check_holder_outgoing():
                    selected_holder = shared_components.get('inbound_holder_select').value
                    if not selected_holder or not is_valid_pubkey(selected_holder):
                        ui.notify("Please select a valid holder from the dropdown.", type='warning')
                        return

                    try:
                        tx_num = int(user_sig_count.value)
                    except ValueError:
                        ui.notify("Invalid signature count. Must be an integer.", type='negative')
                        return

                    holder_total_out, holder_data = await async_check_outgoing_via_shyft(
                        wallet_addr_str=selected_holder, 
                        target_mint_str=sel_token["mint"], 
                        tx_num=tx_num
                    )

                    ui.markdown(f"**Holder Wallet Outgoing** => {float(holder_total_out):,.4f} tokens, # TX => {len(holder_data)}").classes('mt-2')

                    if holder_data:
                        out_df = pd.DataFrame(holder_data, columns=["Signature", "Amount"])
                        out_records = out_df.to_dict(orient='records')
                        ui.table(
                            columns=[
                                {'name': 'Signature', 'label': 'Signature', 'field': 'Signature'},
                                {'name': 'Amount', 'label': 'Amount', 'field': 'Amount'},
                            ],
                            rows=out_records
                        ).classes('w-1/2 mt-2')
                    else:
                        ui.notify("No outgoing transactions found for this holder with the given signature count & mint.", type='info')

                ui.button("Check Holder Wallet Outgoing", on_click=check_holder_outgoing).classes('mt-2')

                ui.markdown("#### Step 4: Arbitrary Wallet => inbound from BC 🔄")
                arbitrary_wallet_input = ui.input(label="Enter a wallet pubkey to see inbound from BC").classes('w-1/2 mt-2')

                async def check_arbitrary_inbound():
                    w = arbitrary_wallet_input.value
                    if not w:
                        ui.notify("Please enter a wallet address", type='warning')
                        return
                    if not is_valid_pubkey(w):
                        ui.notify("Invalid Solana address format.", type='negative')
                        return
                    try:
                        tx_num = int(user_sig_count.value)
                    except ValueError:
                        ui.notify("Invalid signature count. Must be an integer.", type='negative')
                        return

                    bonding_curve = shared_state.get("bc_address") or ""


                    inbound_tot, inbound_data = await async_check_received_from_bonding_curve_via_shyft(
                        holder_addr_str=w,
                        bonding_curve_str=bonding_curve,
                        target_mint_str=sel_token["mint"],
                        tx_num=tx_num
                    )
                    ui.markdown(f"**Inbound** => {float(inbound_tot):,.4f}, # TX => {len(inbound_data)}").classes('mt-2')
                    if inbound_data:
                        inbound_df = pd.DataFrame(inbound_data, columns=["Signature", "Amount"])
                        inbound_records = inbound_df.to_dict(orient='records')
                        ui.table(
                            columns=[
                                {'name': 'Signature', 'label': 'Signature', 'field': 'Signature'},
                                {'name': 'Amount', 'label': 'Amount', 'field': 'Amount'},
                            ],
                            rows=inbound_records,
                        ).classes('w-1/2 mt-2')
                    else:
                        ui.notify("No inbound TX found for that wallet.", type='info')

                ui.button("Check Inbound from BC for Arbitrary Wallet", on_click=check_arbitrary_inbound).classes('mt-2')

                ui.markdown("#### Provide Feedback").classes('mt-4')
                feedback_text = ui.textarea(label="Your feedback:").classes('w-1/2 mt-2')

                async def submit_feedback():
                    feedback = feedback_text.value
                    if feedback:
                        logger.info(f"User Feedback: {feedback}")
                        ui.notify("Thank you for your feedback.", type='positive')
                    else:
                        ui.notify("Feedback is empty.", type='warning')

                ui.button("Submit Feedback", on_click=submit_feedback).classes('mt-2')
                ui.notify("✅ Token Analysis steps loaded.", type='positive')

        await build_analysis_steps()

        async def update_selected_token(symbol):
            nonlocal sel_token, token_info_container, analysis_steps_container
            sel_token = tracked_tokens.get(symbol, {})
            if not sel_token:
                ui.notify(f"Token '{symbol}' not found in tracked tokens.", type='warning')
                return

            token_info_container.clear()
            analysis_steps_container.clear()

            with token_info_container:
                ui.markdown(f"### Analyzing: {sel_token.get('name','N/A')} ({sel_token.get('symbol','N/A')})")
                ui.markdown(f"**Mint Address:** `{sel_token.get('mint','N/A')}`")
                ui.markdown(f"**Dev Wallet (Creator):** `{sel_token.get('creator','N/A')}`")
                ui.markdown(f"**Timestamp:** `{sel_token.get('timestamp','N/A')}`")

            await build_analysis_steps()

###########################################################################
##########All new functions related to the Pump page
###########################################################################
async def fetch_metas_current():
    
    url = "https://frontend-api.pump.fun/metas/current"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Current Metadata: {data}")
                else:
                    print(f"error: Server responds with error code {response.status}")
    except Exception as e:
        print(f"error: {str(e)}")
#similar tokens

async def get_similar_tokens(mint, limit=50):
    base_url = "https://frontend-api.pump.fun/coins/similar"
    params = {
        "mint": mint,
        "limit": limit
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Server responds with error code {response.status}"}
    except Exception as e:
        return {"error": str(e)}
#search for metas 

async def search_metas(meta, include_nsfw=True):
    base_url = "https://frontend-api.pump.fun/metas/search"
    params = {
        "meta": meta,
        "includeNsfw": str(include_nsfw).lower()
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Server responds: {response.status}"}
    except Exception as e:
        return {"error": str(e)}
    
#about to graduate tokens
async def fetch_about_to_graduate():
    url = "https://advanced-api.pump.fun/coins/about-to-graduate"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Server  Errorcode: {response.status}"}
    except Exception as e:
        return {"error": str(e)}
    

#dex paid function 
async def fetch_orders_paid_for(token_address: str):
    chain_id = "solana"
    url = f"https://api.dexscreener.com/orders/v1/{chain_id}/{token_address}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Server responds with error: {response.status}"}
    except Exception as e:
        return {"error": str(e)}
############################################################################
# PAGE 5: PUMPS 
############################################################################
@ui.page('/pump-labs')
async def page_pump_labs():
    ui.add_head_html('''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/tengbao/vanta/dist/vanta.birds.min.js"></script>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        VANTA.BIRDS({
          el: "#vanta-bg",
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0xc1418,
          shininess: 50.00,
          waveHeight: 20.00,
          waveSpeed: 1.00,
          zoom: 1.05
        });
    });
    </script>
    ''')

    ui.add_head_html('''
    <style>
      #vanta-bg {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: -1;
      }
    </style>
    ''')

    ui.html('<div id="vanta-bg"></div>')
    
    
    setup_navigation()
    # Wrap everything in a container with a dark background
    with ui.row().classes('bg-gray-900 text-white max-h-screen'):
        
        # LEFT COLUMN: input box, fetch button
        with ui.column().classes('w-1/3 p-4'):
            ui.label("Pumps: Explore Various Features").classes('text-xl font-bold mb-4')
            
            token_input = ui.input(
                label="Enter a token address to check (for similar tokens & DEX paid)",
                placeholder="e.g. 4k3Dyjzvzp8e1T2X..."
            ).classes('mb-4')

            fetch_button = ui.button("FETCH PUMP DATA").classes('mb-4')

        # RIGHT COLUMN: Data display (the tables, etc.)
        with ui.column().classes('w-2/3 p-4'):
            
            # Inject custom styles for glowing tables
            custom_styles = """
            <style>
            .glow-table {
              border-collapse: collapse;
              width: 100%;
              margin-top: 1rem;
              box-shadow: 0 0 5px rgba(255, 255, 255, 0.3),
                          0 0 10px rgba(255, 255, 255, 0.2);
            }
            .glow-table thead th {
              background-color: rgba(255, 255, 255, 0.1);
            }
            .glow-table tbody tr:hover {
              background-color: rgba(255, 255, 255, 0.1);
            }
            .glow-table td, .glow-table th {
              border: 1px solid rgba(255, 255, 255, 0.2);
              padding: 0.5rem 0.75rem;
              text-align: left;
            }
            </style>
            """
            ui.html(custom_styles)

            # SECTION 1: "Tokens About to Graduate"
            # Create a placeholder that shows a table with "No data yet"
            about_to_grad_html = ui.html('''
            <div>
              <h2 class="text-xl font-bold mb-2">Tokens About to Graduate</h2>
              <table class="glow-table">
                <thead>
                  <tr>
                    <th>Image</th>
                    <th>Coin Mint</th>
                    <th>Dev</th>
                    <th>Name</th>
                    <th>Ticker</th>
                    <th>Market Cap</th>
                    <th>Bonding Curve %</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td colspan="7" style="text-align:center;">No data yet</td>
                  </tr>
                </tbody>
              </table>
            </div>
            ''').classes('mb-8')

            # SECTION 2: "DEX Paid Orders"
            # Another placeholder with an empty glow-table
            dex_paid_html = ui.html('''
            <div>
              <h2 class="text-xl font-bold mb-2">DEX Paid Orders</h2>
              <table class="glow-table">
                <thead>
                  <tr>
                    <th>Order ID</th>
                    <th>Price</th>
                    <th>Quantity</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td colspan="3" style="text-align:center;">No data yet</td>
                  </tr>
                </tbody>
              </table>
            </div>
            ''').classes('mb-8')

            # SECTION 3: "Other" Data in a Quasar Table (NiceGUI's ui.table)
            columns = [
                {"name": "Feature", "label": "Feature", "field": "Feature", "sortable": False},
                {"name": "Result",  "label": "Result (JSON/Text)", "field": "Result", "sortable": False},
            ]
            results_table = ui.table(columns=columns, rows=[]).classes('w-full text-white')

    # =========== EVENT HANDLER FOR FETCH BUTTON ===========
    async def on_fetch_data():
        token_addr = token_input.value.strip()
        if not token_addr:
            ui.notify("Please enter a token address first!", type='warning')
            return
        
        # -----------------------------
        # CALL EACH OF YOUR ASYNC FUNCTIONS (replace with real calls)
        # -----------------------------
        metas_current_data  = await fetch_metas_current()
        similar_tokens_data = await get_similar_tokens(mint=token_addr, limit=10)
        search_data         = await search_metas(meta="meta", include_nsfw=True)
        about_to_grad_data  = await fetch_about_to_graduate()
        dex_paid_data       = await fetch_orders_paid_for(token_addr)

        # 1) Build HTML for "Tokens About to Graduate"
        grad_rows_html = ""
        for token_info in about_to_grad_data:
            grad_rows_html += f"""
            <tr>
              <td>
                <img src="{token_info['imageUrl']}" alt="Token Image" class="h-10 w-10 rounded-full" />
              </td>
              <td>{token_info['coinMint']}</td>
              <td>{token_info['dev']}</td>
              <td>{token_info['name']}</td>
              <td>{token_info['ticker']}</td>
              <td>{token_info['marketCap']}</td>
              <td>{token_info['bondingCurveProgress']}</td>
            </tr>
            """

        about_to_grad_html.content = f"""
        <div>
          <h2 class="text-xl font-bold mb-2">Tokens About to Graduate</h2>
          <table class="glow-table">
            <thead>
              <tr>
                <th>Image</th>
                <th>Coin Mint</th>
                <th>Dev</th>
                <th>Name</th>
                <th>Ticker</th>
                <th>Market Cap</th>
                <th>Bonding Curve %</th>
              </tr>
            </thead>
            <tbody>
              {grad_rows_html}
            </tbody>
          </table>
        </div>
        """

        # 2) Build HTML for "DEX Paid Orders"
        paid_rows_html = ""
        for order_info in dex_paid_data:
            paid_rows_html += f"""
            <tr>
              <td>{order_info.get('orderId', '')}</td>
              <td>{order_info.get('price', '')}</td>
              <td>{order_info.get('quantity', '')}</td>
            </tr>
            """

        # Insert token_addr in the title to clarify the data
        dex_paid_html.content = f"""
        <div>
          <h2 class="text-xl font-bold mb-2">
            DEX Paid Orders for {token_addr}
          </h2>
          <table class="glow-table">
            <thead>
              <tr>
                <th>Order ID</th>
                <th>Price</th>
                <th>Quantity</th>
              </tr>
            </thead>
            <tbody>
              {paid_rows_html if paid_rows_html else 
                '<tr><td colspan="3" style="text-align:center;">No orders found</td></tr>'}
            </tbody>
          </table>
        </div>
        """

        # 3) Update the Quasar table for the "Other" data
        table_rows = [
            {"Feature": "Metas Current",  "Result": str(metas_current_data)},
            {"Feature": "Similar Tokens", "Result": str(similar_tokens_data)},
            {"Feature": "Search Metas",   "Result": str(search_data)},
        ]
        results_table.rows = table_rows

        ui.notify("Data fetched successfully!", type='positive')

    fetch_button.on('click', on_fetch_data)


# Suppose you store the user-defined filters in a global dict in shared_state:
# Suppose you store the user-defined filters in a global dict in shared_state:
BUY_FILTER_KEY = "buy_filter"
SELL_FILTER_KEY = "sell_filter"

app = FastAPI()

# Path to your JSON file
KOH_TOKENS_JSON = "king_of_the_hill_tokens.json"
async def build_tracked_table_html(filtered_tokens: list) -> str:
    """
    Return a big HTML string containing the 'Currently Tracked Tokens' table.
    
    Args:
        filtered_tokens (list): List of token dictionaries after applying filters.
        
    Returns:
        str: HTML string representing the table.
    """
    logger.debug(f"Building table with {len(filtered_tokens)} tokens.")
    
    # Start building the HTML
    table_html = """
    <div class="custom-table-container" style="display: flex; justify-content: center; margin: 20px auto;">
      <div>
        <div class="table-title" style="font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">Currently Tracked Tokens</div>
        <table border="1" cellpadding="5" cellspacing="0" style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Symbol</th>
                    <th>Mint</th>
                    <th>Market Cap (USD)</th>
                    <th>Liquidity (USD)</th>
                    <th>Dev Holding (%)</th>
                    <th>Top 20 (%)</th>
                    <th>Max Holder (%)</th>
                    <th>5m Change</th>
                    <th>1h Change</th>
                    <th>6h Change</th>
                    <th>24h Change</th>
                    <th>Token Image</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
    """

    # Build rows for each token
    for i, token in enumerate(filtered_tokens):
        if not isinstance(token, dict):
            logger.error(f"Token at index {i} is not a dictionary: {token}")
            continue  # Skip invalid entries

        name = token.get("name", "N/A")
        symbol_val = token.get("symbol", "N/A")
        mint = token.get("address", "N/A")
        mc = token.get("usd_market_cap", 0.0)
        liq = token.get("liquidity_usd", 0.0)
        dev_pct = token.get("dev_holding_percent", 0.0)
        top20_pct = token.get("top20_percent", 0.0)
        max_holder_pct = token.get("max_holder_percent", 0.0)
        velocity_m5 = token.get("velocity_m5", 0.0)
        velocity_h1 = token.get("velocity_h1", 0.0)
        velocity_h6 = token.get("velocity_h6", 0.0)
        velocity_h24 = token.get("velocity_h24", 0.0)
        image_uri = token.get("image_uri", "")

        # Asynchronously fetch and encode image
         # Encode image to base64 for embedding
        if image_uri:
            try:
                response = requests.get(image_uri)
                response.raise_for_status()
                image_bytes = response.content
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="{symbol_val} Token" width="50" height="50">'
            except Exception as e:
                logger.error(f"Failed to fetch image for token {symbol_val}: {e}")
                image_html = "No Image"
        else:
            image_html = "No Image"

        # Add Buy and Sell buttons with JavaScript event handlers
        actions = f'''
                <button onclick="buyToken('{symbol_val}')" style="margin-right: 5px;">Buy</button>
                <button onclick="sellToken('{symbol_val}')">Sell</button>
            '''

        # Add row to HTML
        table_html += f"""
        <tr>
            <td>{name}</td>
            <td>{symbol_val}</td>
            <td>{mint}</td>
            <td>${mc:,.2f}</td>
            <td>${liq:,.2f}</td>
            <td>{dev_pct:.2f}%</td>
            <td>{top20_pct:.2f}%</td>
            <td>{max_holder_pct:.2f}%</td>
            <td>{velocity_m5:.2f}</td>
            <td>{velocity_h1:.2f}</td>
            <td>{velocity_h6:.2f}</td>
            <td>{velocity_h24:.2f}</td>
            <td>{image_html}</td>
            <td>{actions}</td>
        </tr>
        """

    # Close out HTML
    table_html += """
            </tbody>
        </table>
      </div>
    </div>
    """

    logger.debug("Table HTML built successfully.")
    return table_html

@app.get("/api/tokens")
async def get_tokens():
    """API endpoint to retrieve tracked tokens."""
    if not os.path.exists(KOH_TOKENS_JSON):
        raise HTTPException(status_code=404, detail="Token data not found.")
    try:
        async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
            content = await f.read()
            tokens = json.loads(content)
            return JSONResponse(content=tokens)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding token data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
# trade_page.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from nicegui import ui, app
import json
import os
import logging
import httpx
import asyncio
import base64

app = FastAPI()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Global shared state and globals
shared_state = SharedState()
filters_applied = False
processed_tokens = set()  # Global set to track processed tokens for auto-trade
KOH_TOKENS_JSON = "king_of_the_hill_tokens.json"
filter_inputs = {}  # Global dict for filter input widgets

# ----- Helper Functions -----



async def update_filters(filter_table_container: ui.column) -> None:
    """
    Uses the global filter_inputs (populated by the filter input widgets) to filter tokens
    from KOH_TOKENS_JSON and then updates the filter_table_container with the filtered tokens.
    (This is your original implementation from surveillance.)
    """
    try:
        min_mc = float(filter_inputs["min_market_cap"].value)
        max_mc = float(filter_inputs["max_market_cap"].value)
        min_vel = float(filter_inputs["min_velocity"].value)
        max_vel = float(filter_inputs["max_velocity"].value)
        min_dev = float(filter_inputs["min_dev_holding"].value)
        max_dev = float(filter_inputs["max_dev_holding"].value)
        min_liq = float(filter_inputs["min_liquidity"].value)
        max_liq = float(filter_inputs["max_liquidity"].value)
        min_tg = float(filter_inputs["min_tg_pop"].value)
        max_tg = float(filter_inputs["max_tg_pop"].value)
        min_top20_pct = float(filter_inputs["min_top20_pct"].value)
        max_top20_pct = float(filter_inputs["max_top20_pct"].value)
        min_vel_5 = float(filter_inputs["min_velocity_5"].value)
        max_vel_5 = float(filter_inputs["max_velocity_5"].value)
        min_vel_15 = float(filter_inputs["min_velocity_1"].value)
        max_vel_15 = float(filter_inputs["max_velocity_1"].value)
        min_vel_60 = float(filter_inputs["min_velocity_6"].value)
        max_vel_60 = float(filter_inputs["max_velocity_6"].value)
        min_vel_720 = float(filter_inputs["min_velocity_24"].value)
        max_vel_720 = float(filter_inputs["max_velocity_24"].value)
        max_single_holder = float(filter_inputs["max_single_holder"].value)
    except ValueError:
        await ui.notify("Please enter valid numeric values for filters", type="error")
        return

    try:
        async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
            tokens = json.loads(await f.read())
    except Exception as e:
        await ui.notify(f"Error reading token data: {e}", type="error")
        return

    filtered_tokens = []
    # For each token, check (for demonstration) market cap and number_of_holders
    for token in tokens:
        mc = token.get("usd_market_cap", 0.0)
        min_holders = int(filter_inputs.get("min_number_of_holders", ui.input(value="0")).value)
        if mc >= min_mc and mc <= max_mc and token.get("number_of_holders", 0) >= min_holders:
            filtered_tokens.append({
                "Name": token.get("name", "N/A"),
                "Symbol": token.get("symbol", "N/A"),
                "Mint": token.get("address", "N/A"),
                "Market Cap (USD)": f"{mc:,.2f}",
                "Liquidity (USD)": f"{token.get('liquidity_usd',0):,.2f}",
                "Dev Holding (%)": f"{token.get('dev_holding_percent',0):.2f}%",
                #"Telegram Popularity": token.get("tg_popularity", 0),
                "Top 20 (%)": f"{token.get('top20_percent',0):.2f}%",
                "Velocity 5-min (%)": f"{token.get('velocity_m5',0):+.2f}%",
                "Velocity 15-min (%)": f"{token.get('velocity_h1',0):+.2f}%",
                "Velocity 60-min (%)": f"{token.get('velocity_h6',0):+.2f}%",
                "Velocity 720-min (%)": f"{token.get('velocity_h24',0):+.2f}%",
                "Token Image": f'<img src="{token.get("image_uri","")}" alt="{token.get("symbol","")} Image" width="50" height="50">' if token.get("image_uri") else "No Image"
            })
    shared_state.set("filtered_tokens", filtered_tokens)
    if filtered_tokens:
        html = '<div style="padding-left:20%;"><h2>Filtered Tokens</h2><table border="1" style="width:80%; border-collapse: collapse;">'
        html += "<thead><tr><th>Name</th><th>Symbol</th><th>Mint</th><th>Market Cap (USD)</th></tr></thead><tbody>"
        for token in filtered_tokens:
            html += f"<tr><td>{token['Name']}</td><td>{token['Symbol']}</td><td>{token['Mint']}</td><td>${token['Market Cap (USD)']}</td></tr>"
        html += "</tbody></table></div>"
    else:
        html = '<div style="padding-left:20%;"><h2>Filtered Tokens</h2><p>No tokens match the filter criteria.</p></div>'
        await ui.notify("No tokens match the filter criteria.", type="warning")
    filter_table_container.clear()
    with filter_table_container:
        ui.html(html)
    ui.update()

# ----- End Helper Functions -----



# -----------------------------------------------------------------------------
# /trade Page Definition
# -----------------------------------------------------------------------------
@ui.page('/trade')
async def page_trade():
    # Set up navigation and background
    setup_navigation()
    global filters_applied, processed_tokens

    # VANTA.js Background and CSS
    ui.add_head_html('''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/tengbao/vanta/dist/vanta.globe.min.js"></script>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        VANTA.GLOBE({
            el: "#vanta-bg",
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            color: 0xffffff,
            backgroundColor: 0x5c5c61
        });
    });
    </script>
    ''')
    ui.add_head_html('''
    <style>
      #vanta-bg {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: -1;
      }
      .filter-section, .auto-sell-filter-section, .trade-section {
          background-color: rgba(255, 255, 255, 0.1);
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
      }
      .btn {
          background-color: #3b82f6;
          color: white;
          padding: 10px 20px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1em;
      }
      .btn:hover {
          background-color: #2563eb;
      }
      .input-field {
          padding: 8px;
          width: 100%;
          box-sizing: border-box;
          margin-bottom: 10px;
          border: 1px solid #ccc;
          border-radius: 4px;
      }
      .glow-table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 20px;
      }
      .glow-table th, .glow-table td {
          border: 1px solid #ddd;
          padding: 8px;
          text-align: center;
      }
      .glow-table th {
          background-color: rgba(255, 255, 255, 0.2);
          color: #000;
      }
      .glow-table tr:nth-child(even) {
          background-color: rgba(255, 255, 255, 0.1);
      }
      .glow-table tr:hover {
          background-color: rgba(255, 255, 255, 0.2);
      }
    </style>
    ''')
    ui.html('<div id="vanta-bg"></div>')
    ui.html('<h1 style="text-align: center; margin-top: 40px; color: white;">Trade Page</h1>')

    # 1. CURRENTLY TRACKED TOKENS TABLE (this section remains unchanged)
    tracked_tokens_container = ui.column()
    

    # 2. MAIN LAYOUT: Two Columns
    with ui.row().classes("justify-between"):
        # LEFT COLUMN: Filter UI and Filtered Tokens Table
        with ui.column().classes("w-1/2"):
            # (A) Auto-Sell Filter Section (if separate from main filters)
            with ui.column().classes("auto-sell-filter-section"):
                ui.markdown("## Auto-Sell Filters").classes("text-center text-2xl font-bold mb-4")
                min_market_cap_auto_sell_input = ui.input(label="Auto-Sell Min Market Cap (USD)", value="0.0").classes("input-field")
                min_holders_auto_sell_input = ui.input(label="Auto-Sell Min Number of Holders", value="0").classes("input-field")
                apply_auto_sell_filter_button = ui.button("Apply Auto-Sell Filter").classes("btn")
            # (B) Filter Section for Auto-Trade & Display
            

        # RIGHT COLUMN: Trade Execution and Auto-Trade Features
        with ui.column().classes("w-1/2"):
            # (A) Trade Execution Section (unchanged)
            with ui.column().classes("trade-section"):
                ui.markdown("## Execute a Trade").classes("text-center text-2xl font-bold mb-4")
                wallet_address_input = ui.input(label="Wallet Address", value="").classes("input-field")
                private_key_input = ui.input(label="Private Key", value="").classes("input-field")
                symbol_dropdown = ui.select(label="Symbol", options=[], value=None).classes("input-field")
                async def load_symbols():
                    if not os.path.exists(KOH_TOKENS_JSON):
                        ui.notify("Token data file not found.", type="negative")
                        return
                    async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
                        try:
                            tokens = json.loads(await f.read())
                        except json.JSONDecodeError:
                            ui.notify("Invalid JSON format.", type="negative")
                            return
                    symbols = [token["symbol"] for token in tokens if "symbol" in token]
                    symbol_dropdown.options = symbols
                    if symbols:
                        symbol_dropdown.value = symbols[0]
                await load_symbols()
                token_address_state = ""
                async def on_symbol_change(value):
                    nonlocal token_address_state
                    async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
                        try:
                            tokens = json.loads(await f.read())
                        except json.JSONDecodeError:
                            ui.notify("Invalid JSON format.", type="negative")
                            return
                    token = next((t for t in tokens if t["symbol"] == value), None)
                    if token and "address" in token:
                        token_address_state = token["address"]
                    else:
                        token_address_state = ""
                    ui.notify(f"Selected Token Address: {token_address_state}", type="info")
                symbol_dropdown.on('change', on_symbol_change)
                action_select = ui.select(label="Action", options=["buy", "sell"], value="buy").classes("input-field")
                amount_input = ui.input(label="Amount", value="").classes("input-field")
                slippage_input = ui.input(label="Slippage (%)", value="10").classes("input-field")
                priority_fee_input = ui.input(label="Priority Fee", value="0.005").classes("input-field")
                execute_trade_button = ui.button("Execute Trade").classes("btn")
                async def execute_trade():
                    try:
                        wallet_address = wallet_address_input.value.strip()
                        private_key = private_key_input.value.strip()
                        symbol = symbol_dropdown.value
                        action = action_select.value
                        amount_str = amount_input.value.strip()
                        slippage_str = slippage_input.value.strip()
                        priority_fee_str = priority_fee_input.value.strip()
                        if not wallet_address or not private_key or not symbol or not amount_str:
                            ui.notify("Wallet Address, Private Key, Symbol, and Amount are required.", type="warning")
                            return
                        try:
                            amount = float(amount_str)
                            slippage = float(slippage_str)
                            priority_fee = float(priority_fee_str)
                        except ValueError:
                            ui.notify("Please enter valid numerical values for Amount, Slippage, and Priority Fee.", type="negative")
                            return
                        async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
                            try:
                                tokens = json.loads(await f.read())
                            except json.JSONDecodeError:
                                ui.notify("Invalid JSON format.", type="negative")
                                return
                        token = next((t for t in tokens if t["symbol"] == symbol), None)
                        if not token:
                            ui.notify("Selected token not found.", type="negative")
                            return
                        token_address = token.get("address")
                        if not token_address:
                            ui.notify("Token address not available.", type="negative")
                            return
                        payload = {
                            "publicKey": wallet_address,
                            "action": action,
                            "mint": token_address,
                            "amount": amount,
                            "denominatedInSol": "true" if action == "buy" else "false",
                            "slippage": slippage,
                            "priorityFee": priority_fee,
                            "pool": "auto"
                        }
                        response = requests.post(url="https://pumpportal.fun/api/trade-local", data=payload)
                        if response.status_code != 200:
                            ui.notify(f"Error while Creating Tx: {response.text}", type="negative")
                            return
                        try:
                            keypair = Keypair.from_base58_string(private_key)
                            transaction = VersionedTransaction.from_bytes(response.content)
                            signed_tx = VersionedTransaction(transaction.message, [keypair])
                        except Exception as e:
                            ui.notify(f"SignatureError: {str(e)}", type="negative")
                            return
                        try:
                            rpc_url = "https://api.mainnet-beta.solana.com/"
                            config = RpcSendTransactionConfig(
                                preflight_commitment=CommitmentLevel.Confirmed,
                                skip_preflight=False
                            )
                            send_payload = SendVersionedTransaction(signed_tx, config).to_json()
                            rpc_response = requests.post(url=rpc_url, headers={"Content-Type": "application/json"}, data=send_payload)
                            if rpc_response.status_code == 200:
                                tx_signature = rpc_response.json().get('result')
                                if tx_signature:
                                    ui.notify(f"Sent Successfully! Tx: https://solscan.io/tx/{tx_signature}", type="positive")
                                else:
                                    ui.notify("RPC Response missing transaction signature.", type="negative")
                            else:
                                ui.notify(f"RPC Error: {rpc_response.text}", type="negative")
                        except Exception as e:
                            ui.notify(f"Send Error: {str(e)}", type="negative")
                    except Exception as e:
                        ui.notify(f"Error executing trade: {e}", type="negative")
                execute_trade_button.on('click', execute_trade)
                def handle_auto_buy_change(e):
                    if e.value:
                        ui.notify("Auto Buy enabled", type="info")
                    else:
                        ui.notify("Auto Buy disabled", type="info")
                def handle_auto_sell_change(e):
                    if e.value:
                        ui.notify("Auto Sell enabled", type="info")
                    else:
                        ui.notify("Auto Sell disabled", type="info")
                with ui.row().classes("items-center mb-2"):
                    ui.label("Auto Buy")
                    auto_buy_switch = ui.switch(value=False, on_change=handle_auto_buy_change)
                with ui.row().classes("items-center mb-4"):
                    ui.label("Auto Sell")
                    auto_sell_switch = ui.switch(value=False, on_change=handle_auto_sell_change)
                # Auto-trade loop: runs every 5 seconds and uses the filtered token list if filters are applied.
                async def auto_trade():
                    global filters_applied, processed_tokens
                    while True:
                        await asyncio.sleep(1)
                        if not filters_applied:
                            continue
                        
                        # Retrieve the filtered tokens list from shared state
                        filtered_tokens = shared_state.get("filtered_tokens")
                        if not filtered_tokens:
                            continue  # No filtered tokens available

                        # Optionally, you might want to re‑validate conditions (if needed)
                        # For this example, we'll assume that the filtered_tokens list
                        # already contains the tokens you want to process for auto-trade.

                        # Process auto-buy for each filtered token:
                        for token in filtered_tokens:
                            symbol = token.get("symbol")
                            # Skip tokens that have already been processed:
                            if symbol in processed_tokens:
                                continue
                            if auto_buy_switch.value:
                                # Ensure required fields are available:
                                if not (wallet_address_input.value.strip() and private_key_input.value.strip() and amount_input.value.strip()):
                                    ui.notify(f"Auto Buy for {symbol} skipped: Missing required fields.", type="warning")
                                    continue
                                try:
                                    amount = float(amount_input.value.strip())
                                    slippage = float(slippage_input.value.strip())
                                    priority_fee = float(priority_fee_input.value.strip())
                                except ValueError:
                                    ui.notify(f"Auto Buy for {symbol} skipped: Invalid numerical values.", type="negative")
                                    continue
                                await perform_auto_trade(
                                    wallet_address=wallet_address_input.value.strip(),
                                    private_key=private_key_input.value.strip(),
                                    symbol=symbol,
                                    action="buy",
                                    amount=amount,
                                    slippage=slippage,
                                    priority_fee=priority_fee
                                )
                                processed_tokens.add(symbol)
                        
                        # Process auto-sell for each filtered token:
                        for token in filtered_tokens:
                            symbol = token.get("symbol")
                            if symbol in processed_tokens:
                                continue
                            if auto_sell_switch.value:
                                if not (wallet_address_input.value.strip() and private_key_input.value.strip() and amount_input.value.strip()):
                                    ui.notify(f"Auto Sell for {symbol} skipped: Missing required fields.", type="warning")
                                    continue
                                try:
                                    amount = float(amount_input.value.strip())
                                    slippage = float(slippage_input.value.strip())
                                    priority_fee = float(priority_fee_input.value.strip())
                                except ValueError:
                                    ui.notify(f"Auto Sell for {symbol} skipped: Invalid numerical values.", type="negative")
                                    continue
                                await perform_auto_trade(
                                    wallet_address=wallet_address_input.value.strip(),
                                    private_key=private_key_input.value.strip(),
                                    symbol=symbol,
                                    action="sell",
                                    amount=amount,
                                    slippage=slippage,
                                    priority_fee=priority_fee
                                )
                                processed_tokens.add(symbol)

                async def perform_auto_trade(wallet_address, private_key, symbol, action, amount, slippage, priority_fee):
                    try:
                        async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
                            try:
                                tokens = json.loads(await f.read())
                            except json.JSONDecodeError:
                                ui.notify(f"Auto {action.capitalize()} Error: Invalid JSON format.", type="negative")
                                return
                        token = next((t for t in tokens if t["symbol"] == symbol), None)
                        if not token:
                            ui.notify(f"Token {symbol} not found for auto {action}.", type="negative")
                            return
                        token_address = token.get("address")
                        if not token_address:
                            ui.notify(f"Token address for {symbol} not available.", type="negative")
                            return
                        payload = {
                            "publicKey": wallet_address,
                            "action": action,
                            "mint": token_address,
                            "amount": amount,
                            "denominatedInSol": "true" if action == "buy" else "false",
                            "slippage": slippage,
                            "priorityFee": priority_fee,
                            "pool": "auto"
                        }
                        response = requests.post(url="https://pumpportal.fun/api/trade-local", data=payload)
                        if response.status_code != 200:
                            ui.notify(f"Auto {action.capitalize()} Error: {response.text}", type="negative")
                            return
                        try:
                            keypair = Keypair.from_base58_string(private_key)
                            transaction = VersionedTransaction.from_bytes(response.content)
                            signed_tx = VersionedTransaction(transaction.message, [keypair])
                        except Exception as e:
                            ui.notify(f"Auto {action.capitalize()} SignatureError: {str(e)}", type="negative")
                            return
                        try:
                            rpc_url = "https://api.mainnet-beta.solana.com/"
                            config = RpcSendTransactionConfig(
                                preflight_commitment=CommitmentLevel.Confirmed,
                                skip_preflight=False
                            )
                            send_payload = SendVersionedTransaction(signed_tx, config).to_json()
                            rpc_response = requests.post(url=rpc_url, headers={"Content-Type": "application/json"}, data=send_payload)
                            if rpc_response.status_code == 200:
                                tx_signature = rpc_response.json().get('result')
                                if tx_signature:
                                    ui.notify(f"Auto {action.capitalize()} Sent! Tx: https://solscan.io/tx/{tx_signature}", type="positive")
                                    if action == "buy":
                                        add_bought_token(token)
                                else:
                                    ui.notify(f"Auto {action.capitalize()} RPC Response missing transaction signature.", type="negative")
                            else:
                                ui.notify(f"Auto {action.capitalize()} RPC Error: {rpc_response.text}", type="negative")
                        except Exception as e:
                            ui.notify(f"Auto {action.capitalize()} Send Error: {str(e)}", type="negative")
                    except Exception as e:
                        ui.notify(f"Auto {action.capitalize()} Error: {e}", type="negative")
                def add_bought_token(token):
                    bought = shared_state.get("bought_tokens") or []
                    bought.append(token)
                    shared_state.set("bought_tokens", bought)
                    asyncio.create_task(update_bought_tokens())
                async def update_bought_tokens():
                    bought = shared_state.get("bought_tokens") or []
                    if bought:
                        html = '<div style="padding-left:20%;"><table border="1" style="width:80%; border-collapse: collapse;"><thead><tr><th>Name</th><th>Symbol</th><th>Mint</th></tr></thead><tbody>'
                        for token in bought:
                            html += f"<tr><td>{token.get('name','')}</td><td>{token.get('symbol','')}</td><td>{token.get('address','')}</td></tr>"
                        html += "</tbody></table></div>"
                    else:
                        html = '<div style="padding-left:20%;"><p>No tokens bought yet.</p></div>'
                    bought_tokens_container.clear()
                    with bought_tokens_container:
                        ui.html(html)
                asyncio.create_task(auto_trade())
            # (B) Bought Tokens Section remains as before
            with ui.column().classes("trade-section"):
                ui.markdown("## Bought Tokens").classes("text-center text-2xl font-bold mb-4")
                bought_tokens_container = ui.column()
                async def update_bought_tokens_table():
                    bought = shared_state.get("bought_tokens") or []
                    if bought:
                        html = '<div style="padding-left:20%;"><table border="1" style="width:80%; border-collapse: collapse;"><thead><tr><th>Name</th><th>Symbol</th><th>Mint</th></tr></thead><tbody>'
                        for token in bought:
                            html += f"<tr><td>{token.get('name','')}</td><td>{token.get('symbol','')}</td><td>{token.get('address','')}</td></tr>"
                        html += "</tbody></table></div>"
                    else:
                        html = '<div style="padding-left:20%;"><p>No tokens bought yet.</p></div>'
                    bought_tokens_container.clear()
                    with bought_tokens_container:
                        ui.html(html)
            # (C) Additional Filter Section for Auto-Trade Settings (already included above)
            with ui.column().classes("white-background").style("padding-left:20%;"):
                ui.markdown("<h2>TRADING SETTINGS (Filters)</h2>")
                with ui.row():
                    slippage_input2 = ui.input(label="Slippage (%)", value="10.0").classes("custom-input")
                    priority_fee_input2 = ui.input(label="Priority Fee (SOL)", value="0.005").classes("custom-input")
                    shared_state.set("slippage", float(slippage_input2.value))
                    shared_state.set("priority_fee", float(priority_fee_input2.value))
                    slippage_input2.on("input", lambda e: shared_state.set("slippage", float(e.value)))
                    priority_fee_input2.on("input", lambda e: shared_state.set("priority_fee", float(e.value)))
                with ui.row():
                    min_market_cap_input = ui.input(label="Min Market Cap (USD)", value="0.0").classes("custom-input")
                    max_market_cap_input = ui.input(label="Max Market Cap (USD)", value="999999.0").classes("custom-input")
                    min_velocity_input = ui.input(label="Min Velocity (%)", value="-100.0").classes("custom-input")
                    max_velocity_input = ui.input(label="Max Velocity (%)", value="1000.0").classes("custom-input")
                with ui.row():
                    min_dev_holding_input = ui.input(label="Min Dev Holding (%)", value="0.0").classes("custom-input")
                    max_dev_holding_input = ui.input(label="Max Dev Holding (%)", value="100.0").classes("custom-input")
                    min_liquidity_input = ui.input(label="Min Liquidity (USD)", value="0.0").classes("custom-input")
                    max_liquidity_input = ui.input(label="Max Liquidity (USD)", value="99999999.0").classes("custom-input")
                with ui.row():
                    min_tg_pop_input = ui.input(label="Min Telegram Popularity", value="0").classes("custom-input")
                    max_tg_pop_input = ui.input(label="Max Telegram Popularity", value="99999").classes("custom-input")
                with ui.row():
                    with ui.column():
                        ui.html('<div class="custom-markdown">Holders Concentration</div>')
                        min_top20_pct_input = ui.input(label="Min Top-20 (%)", value="0.0", placeholder="Minimum top-20 %").classes("custom-input")
                        min_top20_pct_input.tooltip("Minimum percentage held by top 20 holders.")
                    with ui.column():
                        max_top20_pct_input = ui.input(label="Max Top-20 (%)", value="100.0", placeholder="Maximum top-20 %").classes("custom-input")
                        max_top20_pct_input.tooltip("Maximum percentage held by top 20 holders.")
                with ui.row():
                    with ui.column():
                        ui.html('<div class="custom-markdown">Single Holder Filters</div>')
                        max_single_holder_input = ui.input(label="Max Single Holder (%)", value="100.0").classes("custom-input")
                with ui.row():
                    with ui.column():
                        ui.html('<div class="custom-markdown">5-min Price Change Filters</div>')
                        min_velocity_5_input = ui.input(label="Min 5-min Price Change (%)", value="-100.0").classes("custom-input")
                        max_velocity_5_input = ui.input(label="Max 5-min Price Change (%)", value="1000.0").classes("custom-input")
                    with ui.column():
                        ui.html('<div class="custom-markdown">1h Price Change Filters</div>')
                        min_velocity_15_input = ui.input(label="Min 1h Price Change (%)", value="-100.0").classes("custom-input")
                        max_velocity_15_input = ui.input(label="Max 1h Price Change (%)", value="1000.0").classes("custom-input")
                    with ui.column():
                        ui.html('<div class="custom-markdown">6h Price Change Filters</div>')
                        min_velocity_60_input = ui.input(label="Min 6h Price Change (%)", value="-100.0").classes("custom-input")
                        max_velocity_60_input = ui.input(label="Max 6h Price Change (%)", value="1000.0").classes("custom-input")
                    with ui.column():
                        ui.html('<div class="custom-markdown">24h Price Change Filters</div>')
                        min_velocity_720_input = ui.input(label="Min 24h Price Change (%)", value="-100.0").classes("custom-input")
                        max_velocity_720_input = ui.input(label="Max 24h Price Change (%)", value="1000.0").classes("custom-input")
                # Store references to these inputs in the global filter_inputs dictionary
                filter_inputs["min_market_cap"] = min_market_cap_input
                filter_inputs["max_market_cap"] = max_market_cap_input
                filter_inputs["min_velocity"] = min_velocity_input
                filter_inputs["max_velocity"] = max_velocity_input
                filter_inputs["min_dev_holding"] = min_dev_holding_input
                filter_inputs["max_dev_holding"] = max_dev_holding_input
                filter_inputs["min_liquidity"] = min_liquidity_input
                filter_inputs["max_liquidity"] = max_liquidity_input
                filter_inputs["min_tg_pop"] = min_tg_pop_input
                filter_inputs["max_tg_pop"] = max_tg_pop_input
                filter_inputs["min_top20_pct"] = min_top20_pct_input
                filter_inputs["max_top20_pct"] = max_top20_pct_input
                filter_inputs["max_single_holder"] = max_single_holder_input
                filter_inputs["min_velocity_5"] = min_velocity_5_input
                filter_inputs["max_velocity_5"] = max_velocity_5_input
                filter_inputs["min_velocity_1"] = min_velocity_15_input
                filter_inputs["max_velocity_1"] = max_velocity_15_input
                filter_inputs["min_velocity_6"] = min_velocity_60_input
                filter_inputs["max_velocity_6"] = max_velocity_60_input
                filter_inputs["min_velocity_24"] = min_velocity_720_input
                filter_inputs["max_velocity_24"] = max_velocity_720_input
                # Set the flag to indicate if the user has applied filters.
                filters_applied = False

                # Create the container that will display the filtered tokens table.
                filter_table_container = ui.column()

                # Define an async function to handle the button click.
                async def on_apply_filters():
                    global filters_applied
                    filters_applied = True  # Mark that the user has applied filters.
                    await apply_filters(filter_table_container)

                # Create the "Apply Filters" button with the callback.
                ui.button(
                    "Apply Filters", 
                    on_click=lambda: asyncio.create_task(on_apply_filters())
                ).classes("btn")

                # Set up the container placeholder.
                with filter_table_container:
                    ui.html("<div></div>")  # This will be replaced with your filtered table.

                # Create a timer that calls apply_filters every 5 seconds only if filters have been applied.
                ui.timer(5, lambda: asyncio.create_task(apply_filters(filter_table_container)) if filters_applied else None)


    
    
    ui.update()

# -----------------------------------------------------------------------------
# Unified Filtering Function
# -----------------------------------------------------------------------------
async def apply_filters(container):
    global filters_applied
    # Read filter values from the global filter_inputs dictionary
    try:
        min_mc = float(filter_inputs["min_market_cap"].value)
        max_mc = float(filter_inputs["max_market_cap"].value)
        min_vel = float(filter_inputs["min_velocity"].value)
        max_vel = float(filter_inputs["max_velocity"].value)
        min_dev = float(filter_inputs["min_dev_holding"].value)
        max_dev = float(filter_inputs["max_dev_holding"].value)
        min_liq = float(filter_inputs["min_liquidity"].value)
        max_liq = float(filter_inputs["max_liquidity"].value)
        min_tg = float(filter_inputs["min_tg_pop"].value)
        max_tg = float(filter_inputs["max_tg_pop"].value)
        min_top20 = float(filter_inputs["min_top20_pct"].value)
        max_top20 = float(filter_inputs["max_top20_pct"].value)
        max_single = float(filter_inputs["max_single_holder"].value)
        min_vel_5 = float(filter_inputs["min_velocity_5"].value)
        max_vel_5 = float(filter_inputs["max_velocity_5"].value)
        min_vel_1 = float(filter_inputs["min_velocity_1"].value)
        max_vel_1 = float(filter_inputs["max_velocity_1"].value)
        min_vel_6 = float(filter_inputs["min_velocity_6"].value)
        max_vel_6 = float(filter_inputs["max_velocity_6"].value)
        min_vel_24 = float(filter_inputs["min_velocity_24"].value)
        max_vel_24 = float(filter_inputs["max_velocity_24"].value)
    except Exception as e:
        ui.notify("Please ensure all filter values are numeric.", type="negative")
        return

    # Load tokens from the JSON file
    try:
        async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
            content = await f.read()
            tokens = json.loads(content)
    except Exception as e:
        ui.notify(f"Error loading tokens: {e}", type="negative")
        return

    # Assume tokens is a list of token dictionaries
    filtered_tokens = []
    

    for token in tokens:
        try:
            mc = float(token.get("usd_market_cap", 0))
            vel = float(token.get("velocity_m5", 0))
            dev = float(token.get("dev_holding_percent", 0))
            liq = float(token.get("liquidity_usd", 0))
            #tg = float(token.get("tg_popularity", 0))
            top20 = float(token.get("top20_percent", 0))
            max_holder = float(token.get("max_holder_percent", 0))
        except Exception:
            continue
        # Apply all filter criteria
        if (min_mc <= mc <= max_mc and
            min_vel <= vel <= max_vel and
            min_dev <= dev <= max_dev and
            min_liq <= liq <= max_liq and
            #min_tg <= tg <= max_tg and
            min_top20 <= top20 <= max_top20 and
            max_holder <= max_single):
            filtered_tokens.append(token)

    # Build an HTML table from filtered_tokens
    html = "<div style='padding-left:20%;'><table border='1' style='width:80%; border-collapse: collapse;'><thead><tr>"
    headers = ["Name", "Symbol", "Mint", "Market Cap (USD)", "Liquidity (USD)", "Dev Holding (%)", "Top 20 (%)", "5m Change", "1h Change", "6h Change", "24h Change", "Token Image"]
    for header in headers:
        html += f"<th>{header}</th>"
    html += "</tr></thead><tbody>"
    shared_state.set("filtered_tokens", filtered_tokens)
    for token in filtered_tokens:
        html += "<tr>"
        html += f"<td>{token.get('name','N/A')}</td>"
        html += f"<td>{token.get('symbol','N/A')}</td>"
        html += f"<td>{token.get('address','N/A')}</td>"
        html += f"<td>${float(token.get('usd_market_cap',0)):.2f}</td>"
        html += f"<td>${float(token.get('liquidity_usd',0)):.2f}</td>"
        html += f"<td>{float(token.get('dev_holding_percent',0)):.2f}%</td>"
        html += f"<td>{float(token.get('top20_percent',0)):.2f}%</td>"
        html += f"<td>{float(token.get('velocity_m5',0)):.2f}%</td>"
        html += f"<td>{float(token.get('velocity_h1',0)):.2f}%</td>"
        html += f"<td>{float(token.get('velocity_h6',0)):.2f}%</td>"
        html += f"<td>{float(token.get('velocity_h24',0)):.2f}%</td>"
        image_uri = token.get("image_uri", "")
        if image_uri:
            html += f"<td><img src='{image_uri}' alt='{token.get('symbol','Token')}' width='50' height='50'></td>"
        else:
            html += "<td>No Image</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    
    
    # Update the container with the new HTML
    container.clear()
    with container:
        ui.html(html)
    ui.notify("Filters applied.", type="positive")
    filters_applied = True
################### NICEGUI APP ENDS HERE ############################


################### RUN NICEGUI APP ############################
async def refresh_dashboard_data_async():
    # Re-fetch data from your surveillance logic
    #   e.g. update_data() is from surveillance
    from surveillance import update_data, update_tracked_table

    # Actually do the data update
    await update_data()  # re-pulls King-of-the-Hill tokens, DexScreener, etc.

    # Then refresh the table UI
    # If you want to pass a container, do so. Otherwise rely on the 
    # internal logic of update_tracked_table() that uses a global container or param
    # e.g. if you do:
    await update_tracked_table()

    # Optionally re-run filters or do other UI steps
    # ...
    
def refresh_dashboard_data_wrapper():
    asyncio.create_task(refresh_dashboard_data_async())


ui.timer(
    3.0,  # refresh every 3 seconds
    lambda: refresh_dashboard_data_wrapper(),
    active=True
)

#####################pasted here for testing 

if __name__ in {"__main__", "__mp_main__"}:
    #ui.run(on_air=True)
    ui.run(host="0.0.0.0", port=8147, on_air="dlMQrbrYGTHymgPJ")

# Triggered reload at 1737063467.6290057
# Triggered reload at 1737063467.9987113
# Triggered reload at 1737063468.00418
# Triggered reload at 1737063468.008906
# Triggered reload at 1737063468.0694795
# Triggered reload at 1737063469.792337
# Triggered reload at 1737063471.5583584
# Triggered reload at 1737063473.2490292
# Triggered reload at 1737063475.0188444
# Triggered reload at 1737063476.8480506
# Triggered reload at 1737063478.5365913
# Triggered reload at 1737063480.2579978
# Triggered reload at 1737063482.081879
# Triggered reload at 1737063483.7366095
# Triggered reload at 1737063485.3963509
# Triggered reload at 1737063487.1566024
# Triggered reload at 1737063488.9341118
# Triggered reload at 1737063490.7174072
# Triggered reload at 1737063492.8583925
# Triggered reload at 1737063494.908923
# Triggered reload at 1737063496.820526
# Triggered reload at 1737063498.850599
# Triggered reload at 1737063500.6415114
# Triggered reload at 1737063522.552399
# Triggered reload at 1737063822.1251545
# Triggered reload at 1737068025.4017866
# Triggered reload at 1737103157.9953556
# Triggered reload at 1737103408.2226138
# Triggered reload at 1737106814.5862265
# Triggered reload at 1737107124.403303
# Triggered reload at 1737107352.165324
# Triggered reload at 1737279746.665688
# Triggered reload at 1737285210.1035318
# Triggered reload at 1737289509.7013485
# Triggered reload at 1737307366.9631574
# Triggered reload at 1737308663.728154
# Triggered reload at 1737317409.4413283
# Triggered reload at 1737317563.3930166
# Triggered reload at 1737320251.1954844
# Triggered reload at 1737376258.9303174
# Triggered reload at 1737377855.7011182
# Triggered reload at 1737380585.0816872
# Triggered reload at 1737380901.9850469
# Triggered reload at 1737381989.1161942
# Triggered reload at 1737382596.1369765
# Triggered reload at 1737383746.6784606
# Triggered reload at 1737383818.5596473
# Triggered reload at 1737450719.3001978
# Triggered reload at 1737450814.8174298
# Triggered reload at 1737450859.968835
# Triggered reload at 1737450912.5703611
# Triggered reload at 1737465542.646952
# Triggered reload at 1737466292.281122
# Triggered reload at 1737466425.2959902
# Triggered reload at 1737467954.9034526
# Triggered reload at 1737469465.6778984
# Triggered reload at 1737470174.9076757
# Triggered reload at 1737470702.818801
# Triggered reload at 1737470785.0426214
# Triggered reload at 1737470989.1074839
# Triggered reload at 1737477836.1987896
# Triggered reload at 1737477877.1173522