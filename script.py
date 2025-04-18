import os
import openai
from openai import OpenAI
import numpy as np
import random
import json
import re
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# TRUE WORLD STATE
is_rainy = np.random.rand() < 0.30
is_heavy_traffic = np.random.rand() < 0.40

if is_rainy and is_heavy_traffic:
    true_success_prob = 0.30
elif is_rainy and not is_heavy_traffic:
    true_success_prob = 0.50
elif not is_rainy and is_heavy_traffic:
    true_success_prob = 0.55
else:
    true_success_prob = 0.80


def chat_completion(system_text, user_text):
    # client = OpenAI(
    #     base_url="https://models.inference.ai.azure.com",
    #     api_key=os.getenv("GITHUB_TOKEN")
    # )
    
    # response = openai.chat.completions.create(model="gpt-4o",
    # temperature=1.5,
    # messages=[
    #     {"role": "system", "content": system_text},
    #     {"role": "user", "content": user_text},
    # ])
    # return response.choices[0].message.content
    client = ChatCompletionsClient(
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    )

    response = client.complete(
        messages=[
            SystemMessage(system_text),
            UserMessage(user_text),
        ],
        model="gpt-4o",
        temperature=0.0,
        max_tokens=4096,
        top_p=0.1
    )

    return response.choices[0].message.content


# each subagent is prompted for an estimation
def weatherbot_estimate():
    """
    WeatherBot only knows whether 'it might be rainy' or 'it might be clear.'
    We'll feed it partial context. We want a single floating-point probability
    from 0.0 to 1.0 in the LLM's answer.
    """
    weather_context = "rainy" if is_rainy else "clear"
    
    system_text = """You are part of a coordinated team of specialized AI assistants (subagents) working together on a common task: determining whether 
    Route A is a viable option for delivering a package on time. Your role is to contribute a reliable numeric estimate 
    (a probability between 0 and 1) regarding the likelihood of on-time delivery when taking Route A.

    General guidelines:
    1. **Stay In Role**: You have been given a specific perspective or piece of partial information. Only use that information in your estimation.
    2. **Provide a Clear Numeric Estimate**: Your final answer should be a single floating-point number between 0.0 and 1.0, optionally with a very brief explanation. Avoid any extra commentary or unrelated text.
    3. **Be Concise and Focused**: Any reasoning you provide should be minimal. The key output must be the numeric probability.

    Follow these rules and contribute your specialized estimate to help the system evaluate the viability of Route A for package delivery.
    """
    user_text = f"""You are WeatherBot. Your knowledge is limited to the fact that today's weather is likely {weather_context}.
    Estimate the probability that taking Route A will succeed (arrive on time) given this weather.
    Return only a float between 0.0 and 1.0 with minimal explanation.
    """
   
    # Extract the text and parse as float
    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def trafficbot_estimate():
    """
    TrafficBot only knows whether we suspect heavy traffic or light traffic.
    """
    traffic_context = "heavy traffic" if is_heavy_traffic else "light traffic"
    system_text = """You are part of a coordinated team of specialized AI assistants (subagents) working together on a common task: determining whether 
    Route A is a viable option for delivering a package on time. Your role is to contribute a reliable numeric estimate 
    (a probability between 0 and 1) regarding the likelihood of on-time delivery when taking Route A.

    General guidelines:
    1. **Stay In Role**: You have been given a specific perspective or piece of partial information. Only use that information in your estimation.
    2. **Provide a Clear Numeric Estimate**: Your final answer should be a single floating-point number between 0.0 and 1.0, optionally with a very brief explanation. Avoid any extra commentary or unrelated text.
    3. **Be Concise and Focused**: Any reasoning you provide should be minimal. The key output must be the numeric probability.

    Follow these rules and contribute your specialized estimate to help the system evaluate the viability of Route A for package delivery.
    """
    user_text = f"""You are TrafficBot. Your knowledge is limited to the fact that today's traffic is likely {traffic_context}.
    Estimate the probability that taking Route A will succeed (arrive on time) given this traffic situation.
    Return only a float between 0.0 and 1.0.
    """

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def timingbot_estimate():
    """
    TimingBot doesn't know direct weather or traffic,
    but we give it a random "time saving" or "time cost" prompt.
    """
    # Let's feed it a random estimate that "Route A might save 10 minutes" or "might cost 10 minutes."
    # We do this at random for demonstration.
    random_saving = np.random.choice(["save 10 minutes", "cost 10 minutes", "save 5 minutes", "cost 5 minutes"])
    system_text = """You are part of a coordinated team of specialized AI assistants (subagents) working together on a common task: determining whether 
    Route A is a viable option for delivering a package on time. Your role is to contribute a reliable numeric estimate 
    (a probability between 0 and 1) regarding the likelihood of on-time delivery when taking Route A.

    General guidelines:
    1. **Stay In Role**: You have been given a specific perspective or piece of partial information. Only use that information in your estimation.
    2. **Provide a Clear Numeric Estimate**: Your final answer should be a single floating-point number between 0.0 and 1.0, optionally with a very brief explanation. Avoid any extra commentary or unrelated text.
    3. **Be Concise and Focused**: Any reasoning you provide should be minimal. The key output must be the numeric probability.

    Follow these rules and contribute your specialized estimate to help the system evaluate the viability of Route A for package delivery.
    """
    user_text = f"""You are TimingBot. You only know that Route A might {random_saving}.
    Given that partial information, estimate the probability of on-time arrival.
    Return only a float between 0.0 and 1.0.
    """

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def maintenancebot_estimate():
    """
    MaintenanceBot only sees a random 'condition' of Route A: 'smooth' or 'potholes.'
    """
    route_condition = np.random.choice(["potholes", "smooth"])
    system_text = """You are part of a coordinated team of specialized AI assistants (subagents) working together on a common task: determining whether 
    Route A is a viable option for delivering a package on time. Your role is to contribute a reliable numeric estimate 
    (a probability between 0 and 1) regarding the likelihood of on-time delivery when taking Route A.

    General guidelines:
    1. **Stay In Role**: You have been given a specific perspective or piece of partial information. Only use that information in your estimation.
    2. **Provide a Clear Numeric Estimate**: Your final answer should be a single floating-point number between 0.0 and 1.0, optionally with a very brief explanation. Avoid any extra commentary or unrelated text.
    3. **Be Concise and Focused**: Any reasoning you provide should be minimal. The key output must be the numeric probability.

    Follow these rules and contribute your specialized estimate to help the system evaluate the viability of Route A for package delivery.
    """
    user_text = f"""You are MaintenanceBot. You only know that Route A road condition has {route_condition}.
    Based on this partial info, estimate the probability Route A will succeed (arrive on time).
    Return only a float between 0.0 and 1.0.
    """

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

subagent_estimates = {
    "WeatherBot": weatherbot_estimate(),
    "TrafficBot": trafficbot_estimate(),
    "TimingBot": timingbot_estimate(),
    "MaintBot":  maintenancebot_estimate()
}


# now prompt the subagents to state how many shares they are trading based on their estimates
def decide_trade(subagent_name, p_est, market_price, capital):
    """
    The subagent will produce a signed number: positive => buy that many shares,
    negative => short that many shares.
    """
    system_prompt = (
        "You are a simulated agent participating in a research experiment to evaluate the feasibility of using prediction markets "
        "for decision-making. Your task is to assess the likelihood that Route A will successfully deliver a package on time. "
        "You have access to a personal probability estimate and can participate in a toy prediction market by choosing to buy or short shares. "
        "Each share pays out 1 if the event occurs, 0 otherwise. "
        "Make your trade decision based on your estimate, the current market price, and your available capital."
    )

    user_text = (
        f"Agent: {subagent_name}\n"
        f"Your probability estimate for Route A's success: {p_est:.3f}\n"
        f"Current market price: {market_price:.3f}\n"
        f"Available capital: {capital:.2f} units\n"
        "Please decide how many shares to buy (positive) or short (negative) based on this information. "
        "Respond with a brief rationale followed by a single number representing your trade size. "
        "If you choose not to trade, respond with 0."
    )

    response_text = chat_completion(system_prompt, user_text)

    pattern = r"(-?\d+(\.\d+)?)"
    matches = re.findall(pattern, response_text)
    trade_size = 0.0
    if matches:
        trade_str = matches[-1][0]
        try:
            trade_size = float(trade_str)
        except:
            trade_size = 0.0

    print(f"\n--- LLM {subagent_name} decision ---")
    print(f"Prompt:\n{user_text}")
    print(f"Response:\n{response_text}")
    print(f"Parsed trade size: {trade_size}\n")

    return trade_size


# NOW WE ACTUALLY SIMULATE THE MARKET

subagent_capital = {name: 100.0 for name in subagent_estimates}

# initally, set all positions to 0;
# depending on actions (i.e. trade, sell, or nothing)
# move by +1 or -1 or 0 respectively
subagent_positions = {name: 0.0 for name in subagent_estimates}

market_price = 0.50
num_rounds = 5
liquidity = 0.1  # used for simple price movement

for r in range(num_rounds):
    print(f"\n===== ROUND {r+1} =====")
    for name, p_est in subagent_estimates.items():
        capital = subagent_capital[name]
        trade_size = decide_trade(name, p_est, market_price, capital)

        if abs(trade_size) < 1e-9: # too small
            continue

        # handle buying shares
        if trade_size > 0:
            cost = trade_size * market_price
            # check if subagent can afford
            if cost > capital:
                # clip trade_size
                max_shares = int(capital // market_price)
                trade_size = float(max_shares)
                cost = trade_size * market_price
            subagent_capital[name] -= cost
            subagent_positions[name] += trade_size

            # adjust market price
            diff = abs(p_est - market_price)
            market_price += diff * liquidity * (trade_size / 100.0)

        # handle shorting shares
        else:
            # e.g. trade_size = -10 => short 10 shares
            shares_to_short = abs(trade_size)
            if subagent_capital[name] < shares_to_short:
                # can't short that many shares if you can't cover max possible loss
                shares_to_short = int(subagent_capital[name])
                trade_size = -shares_to_short  # reassign for clarity

            credit = shares_to_short * market_price
            subagent_capital[name] += credit
            if subagent_capital[name] < 0:
                # revert if it somehow goes negative
                subagent_capital[name] -= credit
                shares_to_short = 0
                trade_size = 0

            if shares_to_short > 0:
                subagent_positions[name] -= shares_to_short

                # adjust market price
                diff = abs(p_est - market_price)
                market_price -= diff * liquidity * (shares_to_short / 100.0)

    # clip market_price to [0,1]
    market_price = max(0.0, min(1.0, market_price))

    print(f"End of round {r+1}, market_price={market_price:.3f}")

# resolved event based on true probability
did_succeed = (np.random.rand() < true_success_prob)

# payouts are calculated
subagent_payouts = {}
for name in subagent_estimates:
    position = subagent_positions[name]
    final_capital = subagent_capital[name]

    if position > 0:
        # if the event succeeds, each share is worth 1
        if did_succeed:
            gain = position * 1.0
            final_capital += gain
        # if the event fails, shares are worthless
    elif position < 0:
        # short scenario
        short_size = abs(position)
        if did_succeed:
            # must pay short_size * 1.0
            final_capital -= short_size
        # if event fails, they pay 0, keep the credit from earlier
    else:
        # position == 0 => no holdings, no short
        pass

    subagent_payouts[name] = final_capital

# ---------------------------------
# 8. PRINT RESULTS
# ---------------------------------

print("\n=== TRUE WORLD STATE ===")
print(f"Rainy? {is_rainy}, Heavy Traffic? {is_heavy_traffic}")
print(f"True success probability: {true_success_prob:.3f}")
print(f"Route A success? {did_succeed}")

print("\n=== SUBAGENTS' PRIVATE ESTIMATES ===")
for name, p_est in subagent_estimates.items():
    print(f"{name}: {p_est:.3f}")

print(f"\nFinal market price after {num_rounds} rounds: {market_price:.3f}")

print("\n=== FINAL POSITIONS & CAPITAL ===")
for name in subagent_estimates:
    print(f"{name} position: {subagent_positions[name]:.2f}, payout: {subagent_payouts[name]:.2f}")