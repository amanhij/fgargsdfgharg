######################Original working v1#####################


# surveillance_nicegui.py

import time
import pandas as pd
import plotly.express as px
import logging
import random
import re
import os
import json
import aiofiles
from decimal import Decimal
import requests
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from textblob import TextBlob
from aiocache import cached, Cache
import aiohttp
from aiohttp import ClientError
#import tkinter as tk
#from tkinter import simpledialog, messagebox
import threading
import asyncio
from sqlalchemy import select  
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from solders.transaction import VersionedTransaction
from solders.keypair import Keypair
from solders.commitment_config import CommitmentLevel
from solders.rpc.requests import SendVersionedTransaction
from solders.rpc.config import RpcSendTransactionConfig
from datetime import datetime, timedelta, timezone
# Telethon imports
#from telethon import TelegramClient
#from telethon.sessions import StringSession
#from telethon.errors import SessionPasswordNeededError

from shared_state import SharedState
from collections import defaultdict
from typing import List, Dict
# NiceGUI
from nicegui import ui
from nicegui import Client

from copy import deepcopy
# Add the background image as the global style
custom_button_styles = """
<style>
.buy-button {
    background-color: #28a745;
    color: white;
    border: none;
    padding: 5px 10px;
    margin-right: 5px;
    cursor: pointer;
    border-radius: 3px;
}
.buy-button:hover {
    background-color: #218838;
}

.sell-button {
    background-color: #dc3545;
    color: white;
    border: none;
    padding: 5px 10px;
    cursor: pointer;
    border-radius: 3px;
}
.sell-button:hover {
    background-color: #c82333;
}
</style>
"""

# Inject the styles into the application (ideally once at startup)
ui.add_head_html(custom_button_styles)
# Initialize a ThreadPoolExecutor with a suitable number of workers
THREAD_POOL_MAX_WORKERS = 20  # Adjust based on your requirements
global_thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS)
shared_state_lock = asyncio.Lock()
ui.dark_mode()

def format_velocity(value):
    """Formats velocity values with color coding."""
    if value > 0:
        return f'<span style="color:green; font-weight:bold;">{value:+.2f}%</span>'
    elif value < 0:
        return f'<span style="color:red; font-weight:bold;">{value:+.2f}%</span>'
    else:
        return f'<span style="color:gray;">{value:+.2f}%</span>'



async def build_non_moving_tracked_table_html() -> str:
    tracked_tokens = shared_state.get("tracked_tokens") or {}
    ds_changes = shared_state.get("dexscreener_changes") or {}
    #tg_popularity = shared_state.get("tg_popularity") or {}

    table_html = """
    <div class="custom-table-container" style="display: flex; justify-content: flex-start; margin: 20px auto; padding-left: 20%;">
      <div>
        <div class="table-title">Non-moving Tokens (5m Change = 0%)</div>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Symbol</th>
                    <th>Mint</th>
                    <th>Market Cap (USD)</th>
                    <th>Liquidity (USD)</th>
                    <th>Dev Holding (%)</th>
                    <th>Telegram Popularity</th>
                    <th>Top 20 (%)</th>
                    <th>5m Change</th>
                    <th>1h Change</th>
                    <th>6h Change</th>
                    <th>24h Change</th>
                    <th>Token Image</th>
                </tr>
            </thead>
            <tbody>
    """

    # Build rows only for tokens with a 5-minute price change of exactly 0%
    for symbol, data in tracked_tokens.items():
        price_change_info = ds_changes.get(symbol, {})
        m5 = format_velocity(price_change_info.get("m5", 0.0))

        # Filter tokens with 5-minute change exactly 0%
        if price_change_info.get("m5", 0.0) == 0.00:
            name = data.get("name", "N/A")
            symbol_val = data.get("symbol", "N/A")
            mint = data.get("address", "N/A")
            mc = data.get("usd_market_cap", 0.0)
            liq = data.get("liquidity_usd", 0.0)
            dev_pct = data.get("dev_holding_percent", 0.0)
            #pop = tg_popularity.get(symbol, 0)
            top20_pct = data.get("top20_percent", 0.0)
            image_uri = data.get("image_uri", "")
            h1 = format_velocity(price_change_info.get("h1", 0.0))
            h6 = format_velocity(price_change_info.get("h6", 0.0))
            h24 = format_velocity(price_change_info.get("h24", 0.0))

            table_html += f"""
            <tr>
                <td>{name}</td>
                <td>{symbol_val}</td>
                <td>{mint}</td>
                <td>${mc:,.2f}</td>
                <td>${liq:,.2f}</td>
                <td>{dev_pct:.2f}%</td>
                
                <td>{top20_pct:.2f}%</td>
                <td>{m5}</td>
                <td>{h1}</td>
                <td>{h6}</td>
                <td>{h24}</td>
                <td><img src="{image_uri}" alt="{symbol} Token" width="50" height="50"></td>
            </tr>
            """

    # Close out HTML
    table_html += """
            </tbody>
        </table>
      </div>
    </div>
    """

    return table_html

async def execute_buy_ui(client: Client, symbol: str, mint: str):
    user_id = await get_current_user_id(client)
    if user_id is None:
        await safe_notify("User not authenticated. Please log in.", type="error")
        return

    response = await execute_buy(client, {"symbol": symbol, "address": mint})
    if "error" in response:
        await safe_notify(f"Buy failed: {response['error']}", type="error")
    else:
        await safe_notify(f"Buy success: {response['success']}", type="success")

async def execute_sell_ui(client: Client, symbol: str, mint: str):
    user_id = await get_current_user_id(client)
    if user_id is None:
        await safe_notify("User not authenticated. Please log in.", type="error")
        return

    response = await execute_sell(client, {"symbol": symbol, "address": mint})
    if "error" in response:
        await safe_notify(f"Sell failed: {response['error']}", type="error")
    else:
        await safe_notify(f"Sell success: {response['success']}", type="success")



async def build_tracked_table_html(page: int = 1, items_per_page: int = 10) -> str:
    """
    Return a big HTML string containing the entire 'Currently Tracked Tokens' table with pagination.

    Args:
        page (int): The current page number (default is 1).
        items_per_page (int): Number of items to display per page (default is 10).

    Returns:
        str: HTML string for the table and pagination controls.
    """
    # Access whatever data you need from shared_state
    tracked_tokens = shared_state.get("tracked_tokens") or {}
    ds_changes = shared_state.get("dexscreener_changes") or {}

    # Convert tracked_tokens dictionary to a list for easier slicing
    tokens_list = list(tracked_tokens.items())

    # Calculate pagination details
    total_items = len(tokens_list)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    start_index = (page - 1) * items_per_page
    end_index = start_index + items_per_page
    paginated_tokens = tokens_list[start_index:end_index]

    # Start building the HTML
    table_html = """
    <div class="custom-table-container" style="display: flex; justify-content: flex-start; margin: 20px auto; padding-left: 20%;">
      <div>
        <div class="table-title">Currently Tracked Tokens</div>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Symbol</th>
                    <th>Mint</th>
                    <th>Market Cap (USD)</th>
                    <th>Liquidity (USD)</th>
                    <th>Dev Holding (%)</th>
                    <th>Top 20 (%)</th>
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

    # Build rows for each token in the current page
    for symbol, data in paginated_tokens:
        name = data.get("name", "N/A")
        symbol_val = data.get("symbol", "N/A")
        mint = data.get("address", "N/A")
        mc = data.get("usd_market_cap", 0.0)
        liq = data.get("liquidity_usd", 0.0)
        dev_pct = data.get("dev_holding_percent", 0.0)
        top20_pct = data.get("top20_percent", 0.0)
        image_uri = data.get("image_uri", "")

        # Add Buy and Sell buttons with JavaScript event handlers
        actions = f'''
            <button onclick="buyToken('{symbol}')" class="buy-button">Buy</button>
            <button onclick="sellToken('{symbol}')" class="sell-button">Sell</button>
        '''

        # Get price change data
        price_change_info = ds_changes.get(symbol, {})
        if price_change_info.get("m5", 0.0) == 0.00:
            continue
        m5 = format_velocity(price_change_info.get("m5", 0.0))
        h1 = format_velocity(price_change_info.get("h1", 0.0))
        h6 = format_velocity(price_change_info.get("h6", 0.0))
        h24 = format_velocity(price_change_info.get("h24", 0.0))

        table_html += f"""
        <tr>
            <td>{name}</td>
            <td>{symbol_val}</td>
            <td>{mint}</td>
            <td>${mc:,.2f}</td>
            <td>${liq:,.2f}</td>
            <td>{dev_pct:.2f}%</td>
            <td>{top20_pct:.2f}%</td>
            <td>{m5}</td>
            <td>{h1}</td>
            <td>{h6}</td>
            <td>{h24}</td>
            <td><img src="{image_uri}" alt="{symbol} Token" width="50" height="50"></td>
            <td>{actions}</td>
        </tr>
        """

    # Close out HTML table
    table_html += """
            </tbody>
        </table>
      </div>
    </div>
    """

    # Add pagination controls
    pagination_html = """
    <div class="pagination" style="display: flex; justify-content: center; margin-top: 20px;">
    """
    if page > 1:
        pagination_html += f"""
        <button onclick="loadPage({page - 1})" class="pagination-button">Previous</button>
        """
    for p in range(1, total_pages + 1):
        if p == page:
            pagination_html += f"""
            <button onclick="loadPage({p})" class="pagination-button active">{p}</button>
            """
        else:
            pagination_html += f"""
            <button onclick="loadPage({p})" class="pagination-button">{p}</button>
            """
    if page < total_pages:
        pagination_html += f"""
        <button onclick="loadPage({page + 1})" class="pagination-button">Next</button>
        """
    pagination_html += """
    </div>
    """

    # Combine table and pagination HTML
    final_html = table_html + pagination_html

    return final_html



################################################################################
# ---------------------- CONFIGURATION & LOGGING ---------------------- #
################################################################################

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("surveillance.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)






# Telegram credentials
TELEGRAM_API_ID = "28930210"
TELEGRAM_API_HASH = "7f8ebd13da2b2d9bfd0e068efcb0f8ef"
SESSION_FILE = "session.txt"

# Database setup (SQLite with SQLAlchemy)
Base = declarative_base()


class ScrapedData(Base):
    __tablename__ = "scraped_data"

    id = Column(Integer, primary_key=True)
    channel = Column(String)
    word = Column(String)
    count = Column(Integer)
    avg_sentiment = Column(Float)
    timestamp = Column(DateTime)
   
   
# Inside surveillance_nicegui.py

Base = declarative_base()

class UserKey(Base):
    __tablename__ = "user_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True, nullable=False)
    private_key = Column(String, nullable=False)

async def get_user_key(user_id: int, session: AsyncSession) -> UserKey | None:
    from sqlalchemy import select
    stmt = select(UserKey).where(UserKey.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def store_user_key(user_id: int, private_key: str, session: AsyncSession) -> None:
    from sqlalchemy import select
    stmt = select(UserKey).where(UserKey.user_id == user_id)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        row = UserKey(user_id=user_id, private_key=private_key)
        session.add(row)
    else:
        row.private_key = private_key
    await session.commit()

def get_user_state() -> SharedState:
    if not hasattr(app.session, 'user_state'):
        app.session.user_state = SharedState()
    return app.session.user_state


async def get_current_user_id(client: Client) -> int | None:
    """Fetch the current user's ID from their session."""
    user_state = get_user_state()
    return user_state.get("user_id")
DATABASE_URL = "sqlite+aiosqlite:///scraped_data.db"  # Change to your DB URL
engine = create_async_engine(DATABASE_URL, future=True)


async def setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# External API endpoints
API_URL = "https://frontend-api-v2.pump.fun/coins/king-of-the-hill?includeNsfw=True"

SOLANA_RPC_ENDPOINT = "https://rpc.shyft.to?api_key=QUxN68LsKL8OBCRy"
SHYFT_API_KEY = "QUxN68LsKL8OBCRy"
SOLANATRACKER_API_KEY = "e8751231-29f1-4996-b902-d6ab19e92a7d"

# Thresholds
MARKET_CAP_THRESHOLD = 410
INACTIVITY_LIMIT = 3
FETCH_INTERVAL = 5  # in minutes
HISTORY_LENGTH = 20
GRAPH_THRESHOLD = 390
#GLOBAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2000) 
notification_area = ui.column()

################################################################################
# ---------------------- SAFE NOTIFY FUNCTION ---------------------- #
################################################################################

async def safe_notify(message, type="info"):
    with notification_area:
        ui.notify(message, type=type)


################################################################################
# ---------------------- HELPER FUNCTIONS ---------------------- #
################################################################################

SHYFT_API_KEYS = [
    "NOnSRxJXvhyt7_24",
    "C-TkEzDlmTb4F62Y",
    "lr4CQY0GOdgCaLf7",
    "GZA0oOFEUCQeyIUa",
    # ...
]

shyft_index = 0  # global or store in shared_state

def get_next_shyft_endpoint():
    global shyft_index
    # pick the next key
    api_key = SHYFT_API_KEYS[shyft_index]
    # move index forward
    shyft_index = (shyft_index + 1) % len(SHYFT_API_KEYS)
    return f"https://rpc.shyft.to?api_key={api_key}"



async def send_rpc_request(payload, retries=5, backoff_factor=0.5, max_backoff=16):
    global shyft_index
    for attempt in range(retries):
        endpoint = get_next_shyft_endpoint()  # rotate keys on *every* attempt
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # If too many requests, we can switch keys & retry
                        # Or just wait and let next attempt get the next key
                        wait_time = min(backoff_factor * (2 ** attempt), max_backoff)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
        except aiohttp.ClientError as e:
            # e.g. network error, rotate to next key after wait
            await asyncio.sleep(min(backoff_factor * (2 ** attempt), max_backoff))
    logger.error("Max retries in send_rpc_request.")
    return {}


API1_URL = "https://api.shyft.to/sol/v1/wallet/token_balance"


async def get_token_holdings(wallet_addr: str, token_mint: str) -> float:
    """
    Uses Solana RPC to fetch the total holdings of a specific token in a wallet.

    Args:
        wallet_addr (str): The owner's wallet address.
        token_mint (str): The mint address of the token.

    Returns:
        float: The total token balance for the wallet. Returns 0.0 if no balance or an error occurs.
    """
    # Define the payload for the getTokenAccountsByOwner RPC call
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_addr,
            {"mint": token_mint},
            {"encoding": "jsonParsed", "commitment": "finalized"}
        ],
    }

    # Send the RPC request
    response = await send_rpc_request(payload)
    if not response or "result" not in response:
        logger.error(f"Failed RPC call for wallet {wallet_addr} and mint {token_mint}.")
        return 0.0

    result = response["result"]
    token_accounts = result.get("value", [])
    total_balance = 0

    # Sum the balances from all returned token accounts
    for acct in token_accounts:
        try:
            # Navigate through the parsed account data to extract the token amount
            amount_info = acct["account"]["data"]["parsed"]["info"]["tokenAmount"]
            # token amount in smallest units, decimals, etc.
            amount = float(amount_info["uiAmount"]) if amount_info.get("uiAmount") is not None else 0
            total_balance += amount
        except Exception as e:
            logger.error(f"Error parsing token account data: {e}")
            continue

    return total_balance

########################with thread wrapper above function



@cached(ttl=300)
def fetch_king_of_the_hill_tokens_cached():
    """Fetches and caches the list of King of the Hill tokens."""
    return fetch_king_of_the_hill_tokens()


async def fetch_king_of_the_hill_tokens():
    """Asynchronously fetches the list of King of the Hill tokens from API_URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(API_URL, headers={"accept": "application/json"}, timeout=10) as response:
                response.raise_for_status()
                tokens = await response.json()

        # If the response is a single dict, convert to a list
        if isinstance(tokens, dict):
            tokens = [tokens]

        result = []
        for token in tokens:
            symbol = token.get("symbol")
            address = token.get("mint", "N/A")
            creator = token.get("creator", "Unknown") 
            name = token.get("name", "N/A")
            image_uri = token.get("image_uri", "")
            if symbol is not None and address != "N/A":
                result.append({
                    "symbol": symbol,
                    "address": address,
                    "creator": creator,
                    "name": name,
                    "image_uri": image_uri, 
                })
            else:
                logger.warning(
                    f"Skipping token with symbol '{symbol}' due to invalid address '{address}'."
                )
        return result
    except Exception as e:
        logger.error(f"Error fetching tokens: {e}")
        asyncio.create_task(safe_notify(f"Error fetching tokens: {e}", type="error"))
        return []

############################threaded version below 




SOLANATRACKER_API_KEYS = [
    "0f6825c0-3142-4a7c-820d-9169dde22dba",
    "b95ad6ec-0c78-4ac8-b1b5-eb621b7a47be",
    "82df132a-e677-4f58-ba85-939c664d4e42",
    "b885c57d-a32d-4004-a19e-9ad1cc7a1778",
    "ce80dfd9-afa7-4a3c-8d85-6d725bae4fc9",
    "cf701772-825c-4691-ac68-ca4e99a3636e",
    "e699b222-00ad-4ab7-b1d5-a4ae4decadb9",
    "f9862268-f331-4f97-91cf-04b4475ba75f",
    "f45654c0-3880-4c26-b2f4-fe2b4f3cd6c8",
    "4d2aa5cf-9561-4321-a92e-a0cea542c2b4",
    "2f8ac600-237b-4b50-b2ec-6a127f20e950",
    "7de1fae2-59e8-4493-94b1-05f1f6a8e6a6",
    "45e0be88-2d4c-4595-9c31-4e2b0de1682f",
    "10cd1640-772c-4189-905b-492489192e50",
    "382a3878-c573-433c-a0c0-b3bb1327aad2",
    "28c172c4-aedc-445d-aa03-61e98a70300f",
    "008a87ae-6ce9-4c99-b0f1-53f4b8a04210",
    "d205db18-61bc-49c6-a561-47ac72713e65",
    "85526004-d611-4ae6-a3d1-bd4bba95a07f",
    "008c56fe-b0c7-48b1-8649-694aa0abce75",
    "fcf39c29-4658-4589-b806-412ca0b097ba",
    "ce6bda2e-622e-4ebf-bae6-bc8cad35ae5d",
    "114da5c4-9192-4cb8-b392-64fce2e3cfed",
    "22c171fc-b85d-4eaa-8f52-300f6eac6faa",
    "ba66d395-376e-44bd-a7b6-b26e0dffe94a",
    "308cf3fa-a613-49ea-bf86-f0180d9e23d4",
    "83d6f753-0087-4916-ad28-ac5c06586b93",
    "2f2c137c-3019-4055-bec8-5cae06ca37f9",
    "a53615bd-8056-4544-bd6f-e0bb9f174f68",
    "39523df4-1d26-4f84-85e8-0f4e4b0fa929",
    "e29bacbd-3c79-4328-9806-f2b7d94d1f2b",
]


# Optional: Implement a semaphore to limit concurrent requests
SEM_MAX = 100  # Adjust based on your requirements and API limits
semaphore = asyncio.Semaphore(SEM_MAX)

async def fetch_token_info(token_address, max_total_retries=10):
    """
    Fetches token information from Solana Tracker using available API keys.
    
    Args:
        token_address (str): The address of the token.
        max_total_retries (int): Maximum total retries before giving up. Set to None for infinite retries.
    
    Returns:
        dict or None: Token information if successful, else None.
    """
    url = f"https://data.solanatracker.io/tokens/{token_address}"
    max_attempts_per_key = 2
    base_delay = 1  # in seconds
    max_delay = 30  # in seconds
    total_retries = 0

    async with semaphore:
        while True:
            for key_index, api_key in enumerate(SOLANATRACKER_API_KEYS):
                logger.info(f"[{token_address}] Attempting with API Key {key_index + 1}/{len(SOLANATRACKER_API_KEYS)}: {api_key}")

                for attempt in range(1, max_attempts_per_key + 1):
                    try:
                        headers = {
                            "accept": "application/json",
                            "x-api-key": api_key,
                        }
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, headers=headers, timeout=10) as response:
                                if response.status == 429:
                                    raise ClientError(f"429 Too Many Requests for API Key {api_key}")
                                elif response.status == 404:
                                    logger.error(f"[{token_address}] Token not found.")
                                    return None
                                response.raise_for_status()
                                data = await response.json()

                        # Parse the necessary data
                        token_name = data.get("token", {}).get("name", "N/A")
                        pools = data.get("pools", [])
                        liquidity_info = []
                        for pool in pools:
                            market = pool.get("market", "N/A")
                            liquidity_usd = pool.get("liquidity", {}).get("usd", 0)
                            liquidity_info.append({"market": market, "liquidity_usd": liquidity_usd})

                        total_liquidity_usd = sum(pool["liquidity_usd"] for pool in liquidity_info)
                        logger.info(f"[{token_address}] Successfully fetched data using API Key {api_key}")
                        return {
                            "token_name": token_name,
                            "liquidity_info": liquidity_info,
                            "total_liquidity_usd": total_liquidity_usd,
                        }

                    except ClientError as e:
                        logger.warning(f"[{token_address}] Attempt {attempt}/{max_attempts_per_key} with API Key {api_key} failed: {e}")
                        if attempt < max_attempts_per_key:
                            # Exponential backoff with jitter
                            delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1)
                            logger.info(f"[{token_address}] Retrying in {delay:.2f} seconds...")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"[{token_address}] Max retries reached for API Key {api_key}. Switching to next API key.")
                    except asyncio.TimeoutError:
                        logger.warning(f"[{token_address}] Attempt {attempt}/{max_attempts_per_key} timed out with API Key {api_key}")
                        if attempt < max_attempts_per_key:
                            delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1)
                            logger.info(f"[{token_address}] Retrying in {delay:.2f} seconds...")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"[{token_address}] Max retries due to timeout for API Key {api_key}. Switching to next API key.")
                    except Exception as e:
                        logger.error(f"[{token_address}] Unexpected error on attempt {attempt}/{max_attempts_per_key} with API Key {api_key}: {e}")
                        break  # For unexpected errors, break out of the retry loop

                # Increment total retries after exhausting attempts for a key
                total_retries += max_attempts_per_key
                if max_total_retries and total_retries >= max_total_retries:
                    logger.error(f"[{token_address}] Exceeded maximum total retries ({max_total_retries}). Giving up.")
                    return None

            # After cycling through all API keys, check if max_total_retries is reached
            if max_total_retries and total_retries >= max_total_retries:
                logger.error(f"[{token_address}] Exceeded maximum total retries ({max_total_retries}). Giving up.")
                return None

            # Optional: Pause before restarting the API key rotation
            delay_between_rotations = 5  # seconds
            logger.info(f"[{token_address}] All API keys failed. Restarting API key rotation after {delay_between_rotations} seconds...")
            await asyncio.sleep(delay_between_rotations)

####################With thread wrapper from the above 

primary_requests_count = 0
primary_rate_limit = 30
primary_rate_reset = 60  # seconds


async def fetch_token_details(mint):
    """
    Asynchronously fetch token details such as creator, total supply, market cap, and liquidity,
    with rate limit tracking and fallback mechanisms, using SolanaTracker as the first option.
    """
    # Attempt to fetch data from SolanaTracker first
    solana_tracker_url = f"https://data.solanatracker.io/tokens/{mint}"
    headers = {
        "accept": "application/json",
        "x-api-key": SOLANATRACKER_API_KEY,
    }

    liquidity_usd = 0.0
    usd_market_cap = 0.0
    token_name = "N/A"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(solana_tracker_url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                tracker_data = await response.json()

        # Extract token name
        token_info = tracker_data.get("token", {})
        token_name = token_info.get("name", "N/A")

        # Extract pools data
        pools = tracker_data.get("pools", [])

        # Sum liquidity across all pools
        liquidity_usd = sum(pool.get("liquidity", {}).get("usd", 0) for pool in pools)

        # Extract market cap from the first pool, if available
        if pools:
            usd_market_cap = float(pools[0].get("marketCap", {}).get("usd", 0.0))
        else:
            usd_market_cap = 0.0

        logger.info(f"Fetched from SolanaTracker: Token Name {token_name}, Market Cap ${usd_market_cap}, Liquidity ${liquidity_usd}")
    except Exception as st_error:
        logger.warning(f"SolanaTracker API failed for {mint}: {st_error}")
        # Proceed to fallback methods if SolanaTracker fails

    # Define fallback URLs and helper functions for other APIs
    base_url = f"https://advanced-api.pump.fun/coins/metadata/{mint}"
    backup_base_url = f"https://frontend-api-v2.pump.fun/coins/{mint}"
    v1_base_url = f"https://frontend-api.pump.fun/coins/{mint}"

    async def fetch_data(url, track_rate_limit=False):
        global primary_requests_count, primary_rate_limit, primary_rate_reset
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers={"accept": "*/*"}, timeout=10) as response:
                    if track_rate_limit and url.startswith("https://advanced-api.pump.fun"):
                        primary_rate_limit = int(response.headers.get("x-ratelimit-limit", "30"))
                        remaining = int(response.headers.get("x-ratelimit-remaining", "30"))
                        reset = int(response.headers.get("x-ratelimit-reset", "60"))
                        primary_requests_count += 1
                        if remaining <= 0:
                            logger.info("Primary API rate limit reached.")
                            raise aiohttp.ClientError("Rate limit exceeded")
                    response.raise_for_status()
                    try:
                        return await response.json(content_type=None)
                    except aiohttp.ContentTypeError:
                        logger.warning(f"Non-JSON response for URL {url}. Returning None.")
                        return None
            except Exception as e:
                logger.error(f"Error fetching data from {url}: {e}")
                return None

    async def try_sync_variants(base, description=""):
        for sync_value in ["true", "false"]:
            url = f"{base}?sync={sync_value}"
            try:
                logger.info(f"Trying {description} URL with sync={sync_value}: {url}")
                data = await fetch_data(url)
                if data:
                    logger.debug(f"Response from {description} with sync={sync_value}: {json.dumps(data, indent=2)}")
                    return data
            except Exception as e:
                logger.warning(f"{description} with sync={sync_value} failed: {e}")
        return None

    async def get_bonding_curve_progress():
        url = "https://advanced-api.pump.fun/coins/about-to-graduate"
        data = await fetch_data(url)
        if data:
            return {coin["coinMint"]: coin["bondingCurveProgress"] for coin in data}
        return {}

    # Step 1: Check bonding curve progress
    try:
        bonding_curve_data = await get_bonding_curve_progress()
    except Exception as e:
        logger.error(f"Error fetching bonding curve progress: {e}")
        bonding_curve_data = {}

    use_backup_based_on_bonding = bonding_curve_data.get(mint, 0) >= 99

    # Step 2: Decide initial API based on conditions
    if use_backup_based_on_bonding or primary_requests_count >= primary_rate_limit:
        first_choice_url = backup_base_url
        use_track_rate = False
    else:
        first_choice_url = base_url
        use_track_rate = True

    logger.info(f"Trying URL: {first_choice_url}")
    data = await fetch_data(first_choice_url, track_rate_limit=use_track_rate)

    # Step 3: If first API failed, try fallbacks sequentially
    if not data:
        logger.info("First API failed, moving to fallback sequence.")
        fallback_urls = []
        if first_choice_url == base_url:
            fallback_urls = [backup_base_url, v1_base_url]
        elif first_choice_url == backup_base_url:
            fallback_urls = [v1_base_url]

        for url in fallback_urls:
            if url.startswith("https://frontend-api.pump.fun") or url.startswith("https://frontend-api-v2.pump.fun"):
                description = "V1" if "frontend-api.pump.fun" in url else "V2"
                data = await try_sync_variants(url.split("?")[0], description=description)
            else:
                logger.info(f"Trying fallback URL: {url}")
                data = await fetch_data(url)
            if data:
                logger.info(f"Data fetched successfully from fallback URL: {url}")
                break

    if not data:
        logger.error(f"All API endpoints failed for mint {mint}.")
        asyncio.create_task(safe_notify(f"Error fetching token details for {mint}: All endpoints failed.", type="error"))
        return {
            "creator": "Unknown",
            "usd_market_cap": usd_market_cap,
            "liquidity_usd": liquidity_usd
        }

    creator = data.get("creator", "Unknown") if "creator" in data else data.get("dev", "Unknown")
    # Use usd_market_cap from SolanaTracker if it was successfully retrieved
    usd_market_cap = usd_market_cap if usd_market_cap else float(data.get("usd_market_cap", data.get("marketcap", 0)))

    return {
        "creator": creator,
        "usd_market_cap": usd_market_cap,
        "liquidity_usd": liquidity_usd,
    }


################################################################################
# ---------------------- TKINTER AUTH WINDOW ---------------------- #
################################################################################

#def launch_authentication_window():
#    """Launches a Tkinter window for Telegram authentication (phone, code, password)."""

 #   def send_code():
   #     phone = phone_entry.get()
  #      if not phone:
    #        messagebox.showerror("Error", "Please enter your phone number.")
     #       return
      #  try:
       #     loop = asyncio.new_event_loop()
        #    asyncio.set_event_loop(loop)
#
 #           client = TelegramClient(StringSession(), TELEGRAM_API_ID, TELEGRAM_API_HASH)
  #          loop.run_until_complete(client.connect())
#
 #           if not loop.run_until_complete(client.is_user_authorized()):
  #              loop.run_until_complete(client.send_code_request(phone))
   #             messagebox.showinfo("Success", f"Code sent to {phone}. Please enter the code below.")
    #            logger.info(f"Sent authentication code to {phone}.")
     #           shared_state.set("temp_client", client)
      #          send_code_button.config(state="disabled")
       #         authenticate_button.config(state="normal")
        #    else:
         #       messagebox.showinfo("Info", "Telegram client is already authorized.")
          #      shared_state.set("authenticated", True)
#
 #               loop.run_until_complete(client.disconnect())
  #              root.destroy()
   #     except Exception as e:
    #        messagebox.showerror("Error", f"Failed to send code: {e}")
     #       logger.error(f"Failed to send code to {phone}: {e}")
      #      if "client" in locals():
       #         loop.run_until_complete(client.disconnect())
#
 #   def authenticate():
  #      code = code_entry.get()
   #     password = password_entry.get()
    #    client = shared_state.get("temp_client", None)
     #   if not client:
      #      messagebox.showerror("Error", "No client found. Please request the code first.")
       #     return
        #if not code:
         #   messagebox.showerror("Error", "Please enter the code received via Telegram.")
          #  return
#        try:
 #           loop = asyncio.new_event_loop()
  #          asyncio.set_event_loop(loop)
#
 #           loop.run_until_complete(client.sign_in(code=code))
  #          string_session = client.session.save()
   #         with open(SESSION_FILE, "w") as f:
    ##            f.write(string_session)
      #      messagebox.showinfo("Success", "Authentication successful! Session saved.")
     #       shared_state.set("authenticated", True)

#            logger.info("Telegram authentication successful and session saved.")
 #           loop.run_until_complete(client.disconnect())
  #          root.destroy()
#
 #       except SessionPasswordNeededError:
  #          if not password:
   #             messagebox.showerror("Error", "Two-step verification is enabled. Please enter your password.")
    #            return
     #       try:
      #          loop.run_until_complete(client.sign_in(password=password))
       #         string_session = client.session.save()
        #        with open(SESSION_FILE, "w") as f:
         #           f.write(string_session)
          #      messagebox.showinfo("Success", "Authentication successful! Session saved.")
           #     shared_state.set("authenticated", True)
            #    logger.info("Telegram authentication successful and session saved.")
             #   loop.run_until_complete(client.disconnect())
              #  root.destroy()

        #    except Exception as e2:
         #       messagebox.showerror("Error", f"Authentication failed: {e2}")
          #      logger.error(f"Authentication failed: {e2}")
#
 #       except Exception as e:
  #          messagebox.showerror("Error", f"Authentication failed: {e}")
   #         logger.error(f"Authentication failed: {e}")

    #def on_closing():
     #   client = shared_state.get("temp_client", None)
      #  if client:
       #     loop = asyncio.new_event_loop()
        #    asyncio.set_event_loop(loop)
         #   loop.run_until_complete(client.disconnect())
        #root.destroy()

 #v   root = tk.Tk()
  #  root.title("Telegram Authentication")
  #  root.geometry("400x300")
  #  root.resizable(False, False)
   # root.protocol("WM_DELETE_WINDOW", on_closing)

  #  tk.Label(root, text="Step 1: Enter Telegram Phone Number", font=("Helvetica", 12, "bold")).pack(pady=(20, 5))
  #  phone_entry = tk.Entry(root, width=30)
  #  phone_entry.pack(pady=(5, 10))
#
  #  send_code_button = tk.Button(root, text="Send Code", command=send_code, bg="#3498db", fg="white")
  #  send_code_button.pack(pady=(5, 20))

  #  tk.Label(root, text="Step 2: Enter Code and Password", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
   # code_entry = tk.Entry(root, width=30)
   # code_entry.pack(pady=(5, 10))
  #  tk.Label(root, text="Enter Password (if two-step verification):", font=("Helvetica", 10)).pack(pady=(5, 5))
   # password_entry = tk.Entry(root, width=30, show="*")
  #  password_entry.pack(pady=(5, 10))

   # authenticate_button = tk.Button(root, text="Authenticate", command=authenticate, bg="#2ecc71", fg="white")
  #  authenticate_button.pack(pady=(10, 20))
  #  authenticate_button.config(state="disabled")
#
  #  root.mainloop()


# --------------------------------------------
#    COMMENTED OUT: We no longer use these
# --------------------------------------------
# INTERVALS = [5, 60, 360, 1440]  # 5-min, 60-min, 6-hr (360), 24-hr (1440)
# MAX_INTERVAL = max(INTERVALS)


################################################################################
# ---------------------- TOKEN UPDATES & VELOCITY ---------------------- #
################################################################################

async def record_market_cap(symbol: str, market_cap: float):
    now = datetime.now(timezone.utc)
    history = shared_state.get("interval_market_cap_history") or {}
    if symbol not in history:
        history[symbol] = []
    history[symbol].append((now, market_cap))
    shared_state.set("interval_market_cap_history", history)

    
    # --------------------------------------------
    #    COMMENTED OUT: We used to remove older data
    # --------------------------------------------
    # cutoff = now - timedelta(minutes=MAX_INTERVAL)
    # app_state["interval_market_cap_history"][symbol] = [
    #     (ts, cap)
    #     for (ts, cap) in app_state["interval_market_cap_history"][symbol]
    #     if ts >= cutoff
    # ]


# --------------------------------------------
#    COMMENTED OUT: Interval velocity calculation
# --------------------------------------------
# async def calculate_interval_velocities():
#     """
#     Calculates and stores velocity for intervals like 5-min, 1-hr, etc. 
#     """
#     pass  # fully commented out


# --------------------------------------------
#    COMMENTED OUT: Used for the old "Top Movers" table
# --------------------------------------------
# async def get_top_movers_rows():
#     return []


async def get_largest_token_holders_with_total_supply(token_mint, top_n=20):
    """
    Asynchronously fetches the largest token accounts for a given mint and returns:
      1. A list of tuples (owner_address, holder_amount, holder_percent_of_total_supply) for the top N holders.
      2. The total supply (Decimal) of the token.

    This method is still here if you ever want it, but we no longer rely on on-chain supply for final.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [str(token_mint), {"commitment": "finalized"}],
    }
    res = await send_rpc_request(payload)
    if not res or "result" not in res:
        logger.error("Failed to fetch largest token accounts for mint %s.", token_mint)
        return [], Decimal(0)

    largest_accounts = res["result"]["value"][:top_n]
    t_acct_addrs = [acc["address"] for acc in largest_accounts]
    raw_amounts = [Decimal(acc.get("amount", "0")) for acc in largest_accounts]

    batch_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "getMultipleAccounts",
        "params": [t_acct_addrs, {"encoding": "jsonParsed", "commitment": "finalized"}],
    }
    batch_r = await send_rpc_request(batch_payload)
    if not batch_r or "result" not in batch_r:
        logger.error("Failed to fetch owners for the largest token accounts.")
        owners = ["Unknown"] * len(t_acct_addrs)
    else:
        owners_info = batch_r["result"].get("value", [])
        owners = []
        for ai in owners_info:
            if not ai:
                owners.append("Unknown")
            else:
                parsed_d = ai.get("data", {}).get("parsed", {})
                info = parsed_d.get("info", {})
                owners.append(info.get("owner", "Unknown"))

    sup_payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "getTokenSupply",
        "params": [str(token_mint), {"commitment": "finalized"}],
    }
    sup_r = await send_rpc_request(sup_payload)
    if not sup_r or "result" not in sup_r or "value" not in sup_r["result"]:
        logger.error("Failed to fetch token supply for mint %s.", token_mint)
        total_sup = Decimal(0)
        decimals = 0
    else:
        val = sup_r["result"]["value"]
        raw_sup = Decimal(val.get("amount", "0"))
        decimals = val.get("decimals", 0)
        total_sup = raw_sup / (Decimal(10) ** decimals) if decimals > 0 else raw_sup

    holders = []
    for owner, raw_amt in zip(owners, raw_amounts):
        if total_sup > 0:
            adjusted_balance = raw_amt / (Decimal(10) ** decimals)
            pct_of_total = (adjusted_balance / total_sup) * 100
        else:
            adjusted_balance = Decimal(0)
            pct_of_total = Decimal(0)
        holders.append((owner, adjusted_balance, pct_of_total))

    return holders, total_sup

############# with thread below 


# Add synchronous helper

################################################################################
# ---------------------- DEXSCREENER BATCH FETCH FUNCTION --------------------- #
################################################################################
import requests
from concurrent.futures import ThreadPoolExecutor
import asyncio
import random
import time
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize a ThreadPoolExecutor with a suitable number of workers
THREAD_POOL_MAX_WORKERS = 20  # Adjust based on your requirements
global_thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS)

# Initialize a global requests session for connection pooling
global_session = requests.Session()

# Ensure the thread pool is shut down gracefully on exit
atexit.register(lambda: global_thread_pool.shutdown(wait=True))
# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize shared_state with thread-safe structures
shared_state = {
    "dexscreener_data": {}
}
# Create a lock for thread-safe operations


async def fetch_dexscreener_data_in_batches(mint_addresses: list[str]) -> None:
    """
    Fetch DexScreener data for up to 30 Solana mint addresses at once,
    storing results in shared_state['dexscreener_data'][<mint_address>].
    """
    dexscreener_data = shared_state.get("dexscreener_data")

    if dexscreener_data is None:
        dexscreener_data = {}
        shared_state.set("dexscreener_data", dexscreener_data)

    chunk_size = 30
    for i in range(0, len(mint_addresses), chunk_size):
        chunk = mint_addresses[i : i + chunk_size]
        url = "https://api.dexscreener.com/latest/dex/tokens/" + ",".join(chunk)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    json_data = await response.json()
        except Exception as e:
            logger.warning(f"DexScreener batch request failed for chunk={chunk}: {e}")
            continue

        pairs_list = json_data.get("pairs", [])
        if not pairs_list:
            logger.warning(f"No 'pairs' found in DexScreener response for chunk={chunk}")
            continue

        mint_to_pairs = {}
        for pair in pairs_list:
            base_mint = (pair.get("baseToken", {}).get("address", "") or "").lower()
            quote_mint = (pair.get("quoteToken", {}).get("address", "") or "").lower()
            main_mint = base_mint if base_mint else quote_mint
            if not main_mint:
                continue
            if main_mint not in mint_to_pairs:
                mint_to_pairs[main_mint] = []
            mint_to_pairs[main_mint].append(pair)

        for mnt, these_pairs in mint_to_pairs.items():
            chosen = next((p for p in these_pairs if p.get("dexId") == "raydium"), None)
            liquidity_usd = 0.0

            if chosen:
                liquidity_usd = chosen.get("liquidity", {}).get("usd", 0.0)
            else:
                chosen = next((p for p in these_pairs if p.get("dexId") == "pumpfun"), None)
                if chosen:
                    token_mint = mnt
                    try:
                        volume_data = chosen.get("volume", {})
                        liquidity_usd = volume_data.get("h24", 0.0)
                        #liquidity_usd = token_info.get("total_liquidity_usd", 0.0)
                        
                    except Exception as e:
                        logger.error(f"Error fetching token information for {token_mint}: {e}")
                        liquidity_usd = dexscreener_data.get(mnt, {}).get("liquidityUsd", 0.0)

            if not chosen:
                continue

            market_cap = chosen.get("marketCap", 0.0)
            volume_data = chosen.get("volume", {})
            if liquidity_usd == 0.0:
                liquidity_usd = volume_data.get("h24", 0.0)
                

            price_changes = chosen.get("priceChange", {})
            change_m5 = price_changes.get("m5", 0.0)
            change_h1 = price_changes.get("h1", 0.0)
            change_h6 = price_changes.get("h6", 0.0)
            change_h24 = price_changes.get("h24", 0.0)

            price_native = chosen.get("priceNative", None)
            price_usd = chosen.get("priceUsd", None)
            fdv = chosen.get("fdv", 0.0)

            dexscreener_data[mnt] = {
                "marketCap": market_cap,
                "liquidityUsd": liquidity_usd,
                "priceChanges": {
                    "m5": change_m5,
                    "h1": change_h1,
                    "h6": change_h6,
                    "h24": change_h24,
                },
                "priceNative": price_native,
                "priceUsd": price_usd,
                "fdv": fdv,
            }

    shared_state.set("dexscreener_data", dexscreener_data)
    logger.info("DexScreener batch fetch complete.")

###################wITH WRAPPER NESTED BELOW FOR dex BATCH
async def fetch_dexscreener_data_in_batches_threaded(mint_addresses: list[str]) -> None:
    loop = asyncio.get_running_loop()

    def sync_fetch_dex_data():
        # Switch to the default event loop policy in this thread:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(fetch_dexscreener_data_in_batches(mint_addresses))
        finally:
            new_loop.close()

    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, sync_fetch_dex_data)


async def update_data():
    await fetch_king_of_the_hill_tokens_cached.cache.clear()
    new_tokens = await fetch_king_of_the_hill_tokens_cached()
    tracked_tokens = shared_state.get("tracked_tokens")
    logger.info(f"Fetched King-of-the-Hill tokens: {new_tokens}")

    addresses_to_check = {token["address"].lower() for token in new_tokens}
    addresses_to_check.update(details["address"].lower() for details in tracked_tokens.values())
    addresses_to_check = list(addresses_to_check)

    await fetch_dexscreener_data_in_batches_threaded(addresses_to_check)

    dex_changes = shared_state.get("dexscreener_changes") or {}
    if not dex_changes:
        dex_changes = {}
        shared_state.set("dexscreener_changes", dex_changes)

    async def process_token(token):
        symbol = token["symbol"]
        address = token["address"]
        koth_creator = token.get("creator", "Unknown")

        ds_data = (shared_state.get("dexscreener_data") or {}).get(address.lower(), {})
        ds_marketcap = ds_data.get("marketCap", 0.0)
        ds_liquidity = ds_data.get("liquidityUsd", 0.0)
        ds_price_changes = ds_data.get("priceChanges", {})

        if ds_marketcap > 0 and ds_liquidity > 0:
            market_cap = ds_marketcap
            liquidity_usd = ds_liquidity
            dex_changes[symbol] = {
                "m5": ds_price_changes.get("m5", 0.0),
                "h1": ds_price_changes.get("h1", 0.0),
                "h6": ds_price_changes.get("h6", 0.0),
                "h24": ds_price_changes.get("h24", 0.0),
            }
            shared_state.set("dexscreener_changes", dex_changes)
            creator = koth_creator
        else:
            token_details = await fetch_token_details(address)
            fallback_creator = token_details["creator"]
            market_cap = 0
            liquidity_usd =0
            creator = koth_creator if koth_creator != "Unknown" else fallback_creator
            dex_changes.pop(symbol, None)
            #logger.debug(f"Before get_token_holdings: symbol={symbol}, token_mint={address}, creator_wallet={creator}")

        total_supply = 1_000_000_000
        dev_holdings = await get_token_holdings(creator, address)
        dev_holding_percent = (Decimal(dev_holdings) / Decimal(total_supply)) * 100

        holders, _ = await get_largest_token_holders_with_total_supply(address, top_n=20)
        top20_sum = sum(holder[1] for holder in holders)
        top20_pct = (top20_sum / total_supply) * 100 if total_supply > 0 else 0.0
        name = token.get("name", "N/A")
        image_uri = token.get("image_uri", "")
        if holders and total_supply > 0:
            max_holder = max(holders, key=lambda h: h[1])
            max_holder_pct = (max_holder[1] / total_supply) * 100
        else:
            max_holder_pct = 0.0

        token_data = {
            "symbol": symbol,
            "address": address,
            "creator_wallet": creator,
            "usd_market_cap": market_cap,
            "liquidity_usd": liquidity_usd,
            "total_supply": float(total_supply),
            "dev_holdings": float(dev_holdings),
            "dev_holding_percent": float(dev_holding_percent),
            "top20_percent": float(top20_pct),
            "max_holder_percent": float(max_holder_pct),
            "velocity_m5": dex_changes.get(symbol, {}).get("m5", 0.0),
            "velocity_h1": dex_changes.get(symbol, {}).get("h1", 0.0),
            "velocity_h6": dex_changes.get(symbol, {}).get("h6", 0.0),
            "velocity_h24": dex_changes.get(symbol, {}).get("h24", 0.0),
            "name": name,
            "image_uri": image_uri,
        }

        await record_market_cap(symbol, market_cap)

        logger.info(
            f"[{symbol}] => DexScreenerUsed={(ds_marketcap>0 and ds_liquidity>0)}, "
            f"MC=${market_cap:,.2f}, LQ=${liquidity_usd:,.2f}, "
            f"Creator={creator}, Supply={total_supply}, DevHold={dev_holdings}"
        )

        return symbol, token_data

    combined_tokens_dict = {t["symbol"]: t for t in tracked_tokens.values()}
    for t in new_tokens:
        combined_tokens_dict[t["symbol"]] = t
    all_tokens = list(combined_tokens_dict.values())

    tasks = [process_token(token) for token in all_tokens]
    results = await asyncio.gather(*tasks)

    updated_symbols = set()
    for symbol, data in results:
        tracked_tokens[symbol] = data
        updated_symbols.add(symbol)

    tokens_list = list(shared_state.get("tracked_tokens").values())
    await save_tracked_tokens_to_json(tokens_list)

################################################################################
# ---------------------- ASYNC SCRAPER (TELEGRAM) ---------------------- #
################################################################################

#async def scrape_and_update_popularity_async(client):
 #   """
 #   Gets all dialogs/channels, searches last 100 messages for each,
 #   counts mentions of each token's symbol, and updates popularity.
 #   """
 #   tracked_tokens = shared_state.get("tracked_tokens") or {}
 #   tg_popularity = shared_state.get("tg_popularity") or {}
#
  #  dialogs = await client.get_dialogs()
  #  for symbol, details in tracked_tokens.items():
 #       word = symbol
  #      total_count = 0
  #      for dialog in dialogs:
   #         if dialog.is_group or dialog.is_channel:
      #          messages = await client.get_messages(dialog.entity, limit=100)
    #            for message in messages:
   #                 if message.text:
     #                   total_count += len(re.findall(rf"\b{re.escape(word)}\b", message.text, re.IGNORECASE))
     #   popularity = total_count // 1000  # scale factor
     #   tg_popularity[symbol] = int(popularity)
        #logger.info(f"Token {symbol}: scraped count = {total_count}, popularity = {popularity}")
#
 #   shared_state.set("tg_popularity", tg_popularity)


################neested func with thread 
#async def scrape_and_update_popularity(client):
  #  await scrape_and_update_popularity_async(client)

    

################################################################################
# ---------------------- NICEGUI UI SETUP ---------------------- #
################################################################################

filter_inputs = {}

# IMPORTANT: add "field" to each column so Quasar knows which key to use
#ui.html("<h2>Filtered Tokens</h2>")
#filter_table = ui.table(
    
 ##   rows=[],
 #   columns=[
 #       {"name": "Name", "label": "Name", "field": "Name"},
 ##       {"name": "Mint", "label": "Mint", "field": "Mint"},
 #       {"name": "Market Cap (USD)", "label": "Market Cap (USD)", "field": "Market Cap (USD)"},
 #       {"name": "Liquidity (USD)", "label": "Liquidity (USD)", "field": "Liquidity (USD)"},
 #       {"name": "Dev Holding (%)", "label": "Dev Holding (%)", "field": "Dev Holding (%)"},
 #       {"name": "Telegram Popularity", "label": "Telegram Popularity", "field": "Telegram Popularity"},
 #       # We'll leave these velocity columns here if you still want them in the filter table,
        # but they won't actually get updated from the 'interval_velocities' anymore.
##        {"name": "Top 20 (%)", "label": "Top 20 (%)", "field": "Top 20 (%)"},
#        {"name": "Velocity 5-min (%)", "label": "Velocity 5-min (%)", "field": "Velocity 5-min (%)"},
#        {"name": "Velocity 15-min (%)", "label": "Velocity 15-min (%)", "field": "Velocity 15-min (%)"},
#        {"name": "Velocity 60-min (%)", "label": "Velocity 60-min (%)", "field": "Velocity 60-min (%)"},
#        {"name": "Velocity 720-min (%)", "label": "Velocity 720-min (%)", "field": "Velocity 720-min (%)"},
#    ],
#)

chart_placeholder = ui.html("")
# --------------------------------------------
#    COMMENTED OUT: top_movers_table container
# --------------------------------------------
# top_movers_table_container = ui.column()
#tracked_table_container = ui.column()

# Add a new section for Trading Settings

    
async def display_dashboard(filtered_table_container: ui.element):
    """
    Asynchronously builds all the UI elements: headers, tables, and filter fields.
    Populates the filter table within the provided container when filters are applied.
    """
    # Header Buttons
    with ui.row():
        #ui.button("📈 Check Telegram Popularity", on_click=check_telegram_popularity_wrapper)
        ui.button("🔄 Refresh Dashboard", on_click=refresh_dashboard_data_wrapper)

    # Update the Tracked Tokens Table
    await update_tracked_table()

    # Header for Filters Section
    ui.html("<h2>FILTER TOKENS</h2>")



    with ui.column().classes("white-background").style("margin-bottom: 20px;"):
        ui.markdown("<h2>TRADING SETTINGS</h2>")
        with ui.row():
            # Slippage Input
            slippage_input = ui.input(
                label="Slippage (%)",
                value="10.0",  # Default value
                
            ).classes("custom-input")
            
            # Priority Fee Input
            priority_fee_input = ui.input(
                label="Priority Fee (SOL)",
                value="0.005",  # Default value
                
            ).classes("custom-input")

    # Store references to the new inputs in shared_state for global access
    shared_state.set("slippage", float(slippage_input.value))
    shared_state.set("priority_fee", float(priority_fee_input.value))

    # Update shared_state when inputs change
    slippage_input.on("input", lambda e: shared_state.set("slippage", float(e.value)))
    priority_fee_input.on("input", lambda e: shared_state.set("priority_fee", float(e.value)))
    # Row 1: Market Cap and Velocity Inputs
    with ui.row().classes("white-background"):
        min_market_cap_input = ui.input(
            label="Min Market Cap (USD)", value="0.0",
        ).classes("custom-input")
        max_market_cap_input = ui.input(
            label="Max Market Cap (USD)", value="999999.0",
        ).classes("custom-input")
        min_velocity_input = ui.input(
            label="Min Velocity (%)", value="-100.0",
        ).classes("custom-input")
        max_velocity_input = ui.input(
            label="Max Velocity (%)", value="1000.0",
        ).classes("custom-input")

    # Row 2: Dev Holding and Liquidity Inputs
    with ui.row().classes("white-background"):
        min_dev_holding_input = ui.input(
            label="Min Dev Holding (%)", value="0.0",
        ).classes("custom-input")
        max_dev_holding_input = ui.input(
            label="Max Dev Holding (%)", value="100.0",
        ).classes("custom-input")
        min_liquidity_input = ui.input(
            label="Min Liquidity (USD)", value="0.0",
        ).classes("custom-input")
        max_liquidity_input = ui.input(
            label="Max Liquidity (USD)", value="99999999.0",
        ).classes("custom-input")

    # Row 3: Telegram Popularity Inputs
    with ui.row().classes("white-background"):
        min_tg_pop_input = ui.input(
            label="Min Telegram Popularity", value="0",
        ).classes("custom-input")
        max_tg_pop_input = ui.input(
            label="Max Telegram Popularity", value="99999"
        ).classes("custom-input")

    # Row 4: Holders Concentration Inputs
    with ui.row().classes("white-background"):
        with ui.column():
            ui.html('<div class="custom-markdown">Holders Concentration</div>')
            min_top20_pct_input = ui.input(
                label="Min Top-20 (%)", value="0.0", placeholder="Minimum top-20 %"
            ).classes("custom-input")
            min_top20_pct_input.tooltip("Minimum percentage held by top 20 holders.")
        with ui.column():
            max_top20_pct_input = ui.input(
                label="Max Top-20 (%)", value="100.0", placeholder="Maximum top-20 %"
            ).classes("custom-input")
            max_top20_pct_input.tooltip("Maximum percentage held by top 20 holders.")

    # Row 5: Single Holder Filters
    with ui.row().classes("white-background"):
        with ui.column():
            ui.html('<div class="custom-markdown">Single Holder Filters</div>')
            max_single_holder_input = ui.input(
                label="Max Single Holder (%)",
                value="100.0",
                # Removed on_change=update_filters
            ).classes("custom-input")

    # Row 6: Price Change Filters (5-min, 1h, 6h, 24h)
    with ui.row().classes("white-background"):
        with ui.column():
            ui.html('<div class="custom-markdown">5-min Price Change Filters</div>')
            min_velocity_5_input = ui.input(
                label="Min 5-min Price Change (%)",
                value="-100.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
            max_velocity_5_input = ui.input(
                label="Max 5-min Price Change (%)",
                value="1000.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
        with ui.column():
            ui.html('<div class="custom-markdown">1h Price Change Filters</div>')
            min_velocity_15_input = ui.input(
                label="Min 1h Price Change (%)",
                value="-100.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
            max_velocity_15_input = ui.input(
                label="Max 1h Price Change (%)",
                value="1000.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
        with ui.column():
            ui.html('<div class="custom-markdown">6h Price Change Filters</div>')
            min_velocity_60_input = ui.input(
                label="Min 6h Price Change (%)",
                value="-100.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
            max_velocity_60_input = ui.input(
                label="Max 6h Price Change (%)",
                value="1000.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
        with ui.column():
            ui.html('<div class="custom-markdown">24h Price Change Filters</div>')
            min_velocity_720_input = ui.input(
                label="Min 24h Price Change (%)",
                value="-100.0",
                # Removed on_change=update_filters
            ).classes("custom-input")
            max_velocity_720_input = ui.input(
                label="Max 24h Price Change (%)",
                value="1000.0",
                # Removed on_change=update_filters
            ).classes("custom-input")

    # Store references to all filter inputs in the filter_inputs dictionary
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
    
    

    # **Create the "Apply Filters" Button with Proper Event Handler**
    ui.button(
        "Apply Filters",
        on_click=lambda: asyncio.create_task(update_filters(filtered_table_container))
    )

    # Header for Market Cap History
    ui.html("<h2>MARKET CAP HISTORY</h2>")
    await build_chart_section()
  





def format_velocity(value):
    """Formats velocity values with color coding."""
    if value > 0:
        result = f'<span style="color:green; font-weight:bold;">{value:+.2f}%</span>'
    elif value < 0:
        result = f'<span style="color:red; font-weight:bold;">{value:+.2f}%</span>'
    else:
        result = f'<span style="color:gray;">{value:+.2f}%</span>'
    logger.info(f"Formatted velocity: {result}")
    return result

shared_state = SharedState()


async def update_non_moving_table(non_moving_container=None) -> None:
    """
    Updates the 'Non-moving Tokens' (cached tokens) table.
    """
    logger.info("Updating non-moving tracked tokens...")

    async with shared_state_lock:
        # Fetch and build HTML for non-moving tokens
        new_html = await build_non_moving_tracked_table_html()

        # Insert into the UI container if provided
        if non_moving_container is not None:
            with non_moving_container:
                non_moving_container.clear()  # Remove old content
                ui.html(new_html)             # Insert new content

        ui.update()  # Force UI refresh if necessary

    logger.info("Non-moving tokens table updated and inserted.")






async def update_tracked_table(table_container=None) -> None:
    """
    Updates the 'Currently Tracked Tokens' table.
    """
    logger.info("Updating tracked tokens...")

    async with shared_state_lock:
        # Fetch and process data
        new_html = await build_tracked_table_html()

        # Store in app storage if needed
        app.storage.general['tracked_table_html'] = new_html

        # Insert into the UI container if provided
        if table_container is not None:
            with table_container:
                table_container.clear()  # Remove old content
                ui.html(new_html)        # Insert new content

        ui.update()  # Force UI refresh if necessary
    
    logger.info("Tracked tokens table updated and inserted.")


# --------------------------------------------
#    COMMENTED OUT: old "Top Movers" table
# --------------------------------------------
# async def update_top_movers_table():
#     interval_velocities = app_state["interval_velocities"]
#     top_movers_table_container.clear()
#     ...

async def build_chart_section():
    """Asynchronously builds a select box and placeholder for the market cap chart."""
    tracked_tokens = shared_state.get("tracked_tokens")
    valid_for_graph = list(tracked_tokens.keys())
    if not valid_for_graph:
        ui.label("No tokens available for the graph.").classes("text-red-500")
        return

    selected_token = ui.select(
        label="Select a token for history:",
        options=valid_for_graph,
        on_change=lambda: asyncio.create_task(update_chart(selected_token.value)),
    )
    selected_token.value = valid_for_graph[0]
    chart_placeholder.element_id = "chart_html"
    await update_chart(selected_token.value)


async def update_chart(symbol):
    """Asynchronously renders a Plotly line chart of the selected token's market cap over time."""
    market_cap_history = shared_state.get("market_cap_history") or {}
    if symbol not in market_cap_history:
        chart_placeholder.content = "<p class='no-data'></p>"
        return

    caps = market_cap_history[symbol]
    if len(caps) < 2:
        chart_placeholder.content = "<p class='no-data'>Not enough history for this token.</p>"
        return

    slice_caps = caps[-HISTORY_LENGTH:]
    data_plot = []
    for i, c_val in enumerate(slice_caps, start=1):
        data_plot.append({"Iteration": i, "Market Cap (USD)": c_val, "Symbol": symbol})

    df_hist = pd.DataFrame(data_plot)
    fig = px.line(
        df_hist,
        x="Iteration",
        y="Market Cap (USD)",
        color="Symbol",
        title=f"Market Cap Over Time: {symbol}",
    )
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#333"))
    fig_html = fig.to_html(include_plotlyjs="cdn")
    chart_placeholder.content = fig_html



#filter_table_placeholder = None

# Create a styled container and HTML placeholder for Filtered Tokens
#with ui.column() as filter_container:
#    ui.html("")
#    filter_table_placeholder = ui.html("")
#filter_container.classes("custom-table-container")

################################################################################
# ------------------- FILTER & TABLE FOR FILTERED TOKENS ---------------------- #
################################################################################
RPC_URL = "https://api.mainnet-beta.solana.com"
# Add to shared_state initialization
shared_state.set("processed_tokens", set())
shared_state.set("active_positions", {})

# Modified buy function with parameters
async def execute_buy(session: Client, token_info: dict, amount: float = 0.001):
    """Executes a buy order for the specified token for a given user."""
    user_id = await get_current_user_id(session)
    if user_id is None:
        await safe_notify(session, "User not authenticated. Please log in.", type="error")
        return
    async with AsyncSessionLocal() as session:
        row = await get_user_key(user_id, session)
        if not row:
            logger.error(f"No user key found for user_id={user_id}")
            return {"error": f"No user key found for user_id={user_id}"}
    private_key_b58 = row.private_key
    try:
        symbol = token_info["symbol"]
        mint = token_info["address"]
        logger.info(f"Attempting buy for {symbol} ({mint})")
        try:
            kp = Keypair.from_base58_string(private_key_b58)
        except Exception as e:
            logger.error(f"Invalid private key: {e}")
            return {"error": f"Invalid private key: {e}"}
        
        slippage = shared_state.get("slippage", 10.0)  # Default to 10.0 if not set
        priority_fee = shared_state.get("priority_fee", 0.005)  # Default to 0.005 if not set

        # Execute the trade-local API call asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://pumpportal.fun/api/trade-local",
                data={
                    "publicKey": str(kp.pubkey()),
                    "action": "buy",
                    "mint": mint,
                    "amount": amount,
                    "denominatedInSol": "true",
                    "slippage": slippage,
                    "priorityFee": priority_fee,
                    "pool": "pump"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Buy order failed for {symbol}: {error_text}")
                    return {"error": f"Buy order failed: {error_text}"}
                response_content = await response.read()

        # Sign and send transaction
        transaction = VersionedTransaction.from_bytes(response_content)
        signed_tx = VersionedTransaction(transaction.message, [kp])

        config = RpcSendTransactionConfig(
            preflight_commitment=CommitmentLevel.Confirmed,
            skip_preflight=False
        )

        # Execute the RPC request asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=RPC_URL,
                headers={"Content-Type": "application/json"},
                data=SendVersionedTransaction(signed_tx, config).to_json()
            ) as rpc_response:
                if rpc_response.status == 200:
                    rpc_data = await rpc_response.json()
                    tx_signature = rpc_data.get('result')
                    if tx_signature:
                        logger.info(f"Bought {symbol}: https://solscan.io/tx/{tx_signature}")
                        
                        # Track position
                        async with shared_state_lock:
                            active_positions = shared_state.get("active_positions", {})
                            active_positions[mint] = {
                                "entry_time": datetime.now(timezone.utc),
                                "symbol": symbol,
                                "amount": amount,
                                "status": "open"
                            }
                            shared_state.set("active_positions", active_positions)
                        return {"success": f"Bought {symbol}: https://solscan.io/tx/{tx_signature}"}
                    else:
                        logger.error(f"No transaction signature returned for {symbol}.")
                        return {"error": "No transaction signature returned."}
                else:
                    error_text = await rpc_response.text()
                    logger.error(f"RPC request failed for {symbol}: {error_text}")
                    return {"error": f"RPC request failed: {error_text}"}

    except Exception as e:
        logger.error(f"Buy execution failed for {mint}: {str(e)}")
        return {"error": f"Buy execution failed: {str(e)}"}

async def execute_sell(session: Client, token_info: dict, amount: float = 0.001):
    """Executes a sell order for the specified token for a given user."""
    user_id = await get_current_user_id(session)
    if user_id is None:
        await safe_notify(session, "User not authenticated. Please log in.", type="error")
        return
    async with AsyncSessionLocal() as session:
        row = await get_user_key(user_id, session)
        if not row:
            logger.error(f"No user key found for user_id={user_id}")
            return {"error": f"No user key found for user_id={user_id}"}
    
    private_key_b58 = row.private_key
    try:
        symbol = token_info["symbol"]
        mint = token_info["address"]
        logger.info(f"Attempting sell for {symbol} ({mint})")
        try:
            kp = Keypair.from_base58_string(private_key_b58)
        except Exception as e:
            logger.error(f"Invalid private key: {e}")
            return {"error": f"Invalid private key: {e}"}
        
        # Execute the trade-local API call asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://pumpportal.fun/api/trade-local",
                data={
                    "publicKey": str(kp.pubkey()),
                    "action": "sell",  # Change action to 'sell'
                    "mint": mint,
                    "amount": amount,
                    "denominatedInSol": "true",
                    "slippage": 10,
                    "priorityFee": 0.003,
                    "pool": "pump"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Sell order failed for {symbol}: {error_text}")
                    return {"error": f"Sell order failed: {error_text}"}
                response_content = await response.read()
    
        # Sign and send transaction
        transaction = VersionedTransaction.from_bytes(response_content)
        signed_tx = VersionedTransaction(transaction.message, [kp])
    
        config = RpcSendTransactionConfig(
            preflight_commitment=CommitmentLevel.Confirmed,
            skip_preflight=False
        )
    
        # Execute the RPC request asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=RPC_URL,
                headers={"Content-Type": "application/json"},
                data=SendVersionedTransaction(signed_tx, config).to_json()
            ) as rpc_response:
                if rpc_response.status == 200:
                    rpc_data = await rpc_response.json()
                    tx_signature = rpc_data.get('result')
                    if tx_signature:
                        logger.info(f"Sold {symbol}: https://solscan.io/tx/{tx_signature}")
                        
                        # Track position (if applicable)
                        async with shared_state_lock:
                            active_positions = shared_state.get("active_positions", {})
                            position = active_positions.get(mint)
                            if position:
                                position["status"] = "closed"
                                active_positions[mint] = position
                                shared_state.set("active_positions", active_positions)
                        return {"success": f"Sold {symbol}: https://solscan.io/tx/{tx_signature}"}
                    else:
                        logger.error(f"No transaction signature returned for {symbol}.")
                        return {"error": "No transaction signature returned."}
                else:
                    error_text = await rpc_response.text()
                    logger.error(f"RPC request failed for {symbol}: {error_text}")
                    return {"error": f"RPC request failed: {error_text}"}
    
    except Exception as e:
        logger.error(f"Sell execution failed for {mint}: {str(e)}")
        return {"error": f"Sell execution failed: {str(e)}"}


KOH_TOKENS_JSON = "king_of_the_hill_tokens.json"

async def load_tracked_tokens_from_json() -> List[Dict]:
    """Load tracked tokens from the JSON file."""
    try:
        async with aiofiles.open(KOH_TOKENS_JSON, "r") as f:
            content = await f.read()
            tokens = json.loads(content)
            logger.info(f"Loaded {len(tokens)} tokens from {KOH_TOKENS_JSON}.")
            return tokens
    except FileNotFoundError:
        logger.warning(f"{KOH_TOKENS_JSON} not found. Starting with an empty token list.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {KOH_TOKENS_JSON}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading tokens from JSON: {e}")
        return []


    
    
    # Render the filter table HTML using NiceGUI's ui.html()
    # No need to set content separately

    # Render the filter table HTML using NiceGUI's ui.html()
    #filter_table_placeholder.set_content(table_html)

    
################################################################################
# ---------------------- ASYNC BUTTON CALLBACKS ---------------------- #
################################################################################

#async def check_telegram_popularity_async():
 #   """
  #  Checks if we have a valid (authenticated) Telegram session.
   # If not, spawns the Tkinter phone-auth window in another thread.
    #Then scrapes popularity and notifies the user.
   # """
   # try:
   #     if not (shared_state.get("authenticated") or False):#

       #     auth_thread = threading.Thread(target=launch_authentication_window)
      #      auth_thread.start()
       #     auth_thread.join()
#
     #       if os.path.exists(SESSION_FILE):
     #           client = TelegramClient(
     #               StringSession(open(SESSION_FILE).read().strip()),
     ##               TELEGRAM_API_ID,
     #               TELEGRAM_API_HASH,
     #           )
      #          await client.connect()
     #           if not await client.is_user_authorized():
     #               await safe_notify("Telegram client is not authorized. Please try again.", type="warning")
     #               shared_state.set("authenticated", False)

     #           else:
      #              shared_state.set("authenticated", True)
#
       #             logger.info("Telegram client re-initialized and authorized.")
      #              await scrape_and_update_popularity_async(client)
      #              await safe_notify("Telegram popularity updated successfully!", type="success")
     #           await client.disconnect()
     #   else:
      #      if os.path.exists(SESSION_FILE):
      #          client = TelegramClient(
      #              StringSession(open(SESSION_FILE).read().strip()),
     #               TELEGRAM_API_ID,
        #            TELEGRAM_API_HASH,
        #        )
        #        await client.connect()
         #       if not await client.is_user_authorized():
         #           await safe_notify("Telegram client is not authorized. Please authenticate again.", type="warning")
         #           shared_state.set("authenticated", False)

            #    else:
          #          await scrape_and_update_popularity_async(client)
          # #         await safe_notify("Telegram popularity updated successfully!", type="success")
           #     await client.disconnect()
         #   else:
         #       await safe_notify("No Telegram session file. Please authenticate first.", type="warning")

 #   except Exception as e:
    #    await safe_notify(f"Error reconnecting Telegram client: {e}", type="error")
     #   logger.error(f"Error reconnecting Telegram client: {e}")


async def refresh_dashboard_data_async():
    """
    Re-fetch King-of-the-Hill tokens, recalc data, etc.
    Then re-run table-building code.
    """
    await safe_notify("Refreshing dashboard data...", type="info")
    await update_data()
    logger.info("Data updated. Refreshing UI...")
    await update_tracked_table()
    logger.info("Tracked tokens table updated.")

    # --------------------------------------------
    #    COMMENTED OUT: update_top_movers_table
    # --------------------------------------------
    # await update_top_movers_table()

    

    #if (shared_state.get("authenticated") or False) and os.path.exists(SESSION_FILE):
     #   try:
     #       client = TelegramClient(
      #          StringSession(open(SESSION_FILE).read().strip()),
      #          TELEGRAM_API_ID,
      #          TELEGRAM_API_HASH,
      #      )
       #     await client.connect()
       #     # Additional Telegram logic if desired
       #     await client.disconnect()
        #except Exception as e:
       #     await safe_notify(f"Error connecting Telegram client: {e}", type="error")
        #    logger.error(f"Error connecting Telegram client: {e}")

    await safe_notify("Dashboard refresh complete!", type="success")


#def check_telegram_popularity_wrapper():
   # asyncio.create_task(check_telegram_popularity_async())


def refresh_dashboard_data_wrapper():
    logger.info("Refreshing dashboard...")
    asyncio.create_task(refresh_dashboard_data_async())

ui.timer(3, lambda: refresh_dashboard_data_wrapper(), active=True)

import aiofiles
# Define the output JSON file path
KOH_TOKENS_JSON = "king_of_the_hill_tokens.json"

async def save_tracked_tokens_to_json(tokens):
    """Save the list of tracked tokens to a JSON file."""
    try:
        async with aiofiles.open(KOH_TOKENS_JSON, "w") as f:
            await f.write(json.dumps(tokens, indent=4))
        logger.info("Tracked tokens saved to JSON.")
    except Exception as e:
        logger.error(f"Error saving tracked tokens to JSON: {e}")
################################################################################
# ---------------------- STARTUP / MAIN ENTRY POINT ---------------------- #
################################################################################

#async def init_telegram_session():
 #   """
  #  Checks if we have a valid session file, attempts to connect, and sets
  #  app_state["authenticated"] if authorized.
  #  """
 #   if os.path.exists(SESSION_FILE):
 #       try:
  #          client = TelegramClient(
  #              StringSession(open(SESSION_FILE).read().strip()),
   #             TELEGRAM_API_ID,
   #             TELEGRAM_API_HASH,
   #         )
   ##         await client.connect()
    #        if not await client.is_user_authorized():
    #            logger.warning("Session file exists but is not authorized.")
     #           shared_state.set("authenticated", False)

     #       else:
       #         shared_state.set("authenticated", True)

       #         logger.info("Telegram client loaded from session and is authorized.")
       #     await client.disconnect()
        #except Exception as e:
       #     logger.error(f"Error connecting Telegram client at startup: {e}")
    #        shared_state.set("authenticated", False)
#
  #  else:
       # shared_state.set("authenticated", False)

     #   logger.info("No Telegram session file found; not authenticated.")
#

from nicegui import ui, app


#async def run_surveillance_dashboard_async():
#    await setup_database()            # Asynchronously create DB schema
       # Initialize Telegram session
#    await update_data()               # Fetch initial data


#@app.on_startup
#async def on_startup():
#    await display_dashboard()         # Build the initial dashboard UI

async def init_locks():
    app.shared_state_lock = asyncio.Lock()


app.on_startup(init_locks)

    # 1) create the tables if they don’t exist:
    
#def run_surveillance_dashboard():
    # Run asynchronous initialization steps before starting NiceGUI
 #   asyncio.run(run_surveillance_dashboard_async())

    # Schedule periodic refresh tasks using NiceGUI's timer
  #  ui.timer(3, lambda: asyncio.create_task(refresh_dashboard_data_wrapper()))

    # Start the NiceGUI application (starts its own event loop)
    #ui.run(
    #    title="Token Surveillance Dashboard",
    #    dark=False,
    #)


#if __name__ in {"__main__", "__mp_main__"}:
 #   run_surveillance_dashboard()
