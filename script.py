import os
import openai
from openai import OpenAI
import numpy as np
import random
import json
import re
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------
# 1. SET TRUE WORLD STATE
# ---------------------------------
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


# ---------------------------------
# 2. CHAT COMPLETION HELPER
# ---------------------------------
def chat_completion(system_text, user_text):
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    
    response = openai.chat.completions.create(model="gpt-4o",
    temperature=0.0,
    messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ])
    return response.choices[0].message.content


# ---------------------------------
# 3. SUBAGENT ESTIMATES
# ---------------------------------

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
    # naive parse (assuming the LLM returns plain float in text):
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
    # naive parse (assuming the LLM returns plain float in text):
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
    # naive parse (assuming the LLM returns plain float in text):
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def maintenancebot_estimate():
    """
    MaintenanceBot only sees a random 'condition' of Route A: 'smooth' or 'potholes.'
    """
    # We'll do a small random chance of 'potholes' vs 'smooth'
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
    # naive parse (assuming the LLM returns plain float in text):
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


# ---------------------------------
# 4. DECIDE_TRADE FUNCTION (EDITED)
# ---------------------------------
def decide_trade(subagent_name, p_est, market_price, capital):
    """
    The subagent will produce a signed number: positive => buy that many shares,
    negative => short that many shares.
    """
    system_prompt = (
        "You are part of a coordinated team of specialized AI assistants (subagents) working together on a common task: "
        "determining whether Route A is a viable option for delivering a package on time. "
        "You are participating in a prediction market for the likelihood of this outcome. "
        "You have a personal probability estimate for the event. "
        "You can buy shares if you think the event is underpriced, or short shares if you think it's overpriced. "
        "Payout for a share is 1 if the event happens, 0 otherwise. "
        "Make your decision based on your estimate, the market price, and your available capital."
    )

    user_text = (
        f"You are {subagent_name}.\n"
        f"Your private estimate of Route A's success is {p_est:.3f}.\n"
        f"The current market price is {market_price:.3f}.\n"
        f"You have {capital:.2f} units of capital.\n"
        "Question: How many shares would you buy (positive number) or short (negative number)? "
        "Return your final numeric decision in plain text. If 0, do nothing.\n"
        "No extra text. Just your final reasoning (brief) then the numeric trade size."
    )

    response_text = chat_completion(system_prompt, user_text)

    # Parse the final numeric decision from the text
    pattern = r"(-?\d+(\.\d+)?)"
    matches = re.findall(pattern, response_text)
    trade_size = 0.0
    if matches:
        # matches is a list of tuples, each with the numeric string as first item
        trade_str = matches[-1][0]  # last numeric match
        try:
            trade_size = float(trade_str)
        except:
            trade_size = 0.0

    # Debug print
    print(f"\n--- LLM {subagent_name} decision ---")
    print(f"Prompt:\n{user_text}")
    print(f"Response:\n{response_text}")
    print(f"Parsed trade size: {trade_size}\n")

    return trade_size


# ---------------------------------
# 5. MARKET SIMULATION
# ---------------------------------

# Each subagent has initial capital
subagent_capital = {name: 100.0 for name in subagent_estimates}

# We'll track a single net position (positive => long, negative => short)
subagent_positions = {name: 0.0 for name in subagent_estimates}

market_price = 0.50
num_rounds = 3
liquidity = 0.1  # used for simple price movement

for r in range(num_rounds):
    print(f"\n===== ROUND {r+1} =====")
    for name, p_est in subagent_estimates.items():
        capital = subagent_capital[name]
        trade_size = decide_trade(name, p_est, market_price, capital)

        if abs(trade_size) < 1e-9:
            # No trade
            continue

        # BUY scenario: trade_size > 0
        # cost = shares * market_price
        if trade_size > 0:
            cost = trade_size * market_price
            # check if subagent can afford
            if cost > capital:
                # clamp trade_size
                max_shares = int(capital // market_price)
                trade_size = float(max_shares)
                cost = trade_size * market_price
            subagent_capital[name] -= cost
            subagent_positions[name] += trade_size

            # Move market price (toy update)
            diff = abs(p_est - market_price)
            market_price += diff * liquidity * (trade_size / 100.0)

        # SHORT scenario: trade_size < 0
        else:
            # e.g. trade_size = -10 => short 10 shares
            shares_to_short = abs(trade_size)
            # In real markets, you'd credit the subagent some money for shorting at price,
            # but also require margin. We'll do a simple approach:
            # The subagent *receives* shares_to_short * market_price now,
            # but must ensure enough capital to pay if the event occurs (which costs them shares_to_short * 1).
            if subagent_capital[name] < shares_to_short:
                # can't short that many shares if you can't cover max possible loss
                shares_to_short = int(subagent_capital[name])
                trade_size = -shares_to_short  # reassign for clarity

            credit = shares_to_short * market_price
            subagent_capital[name] += credit
            # but effectively we "lock" an amount equal to shares_to_short in capital
            # to ensure they can pay if the event does happen. We'll keep it simple:
            # If they don't have enough capital left to remain safe, clamp further
            if subagent_capital[name] < 0:
                # revert if it somehow goes negative
                subagent_capital[name] -= credit
                shares_to_short = 0
                trade_size = 0

            # final update if short is successful
            if shares_to_short > 0:
                subagent_positions[name] -= shares_to_short

                # toy market price update
                diff = abs(p_est - market_price)
                market_price -= diff * liquidity * (shares_to_short / 100.0)

    # clamp market_price to [0,1]
    market_price = max(0.0, min(1.0, market_price))

    print(f"End of round {r+1}, market_price={market_price:.3f}")


# ---------------------------------
# 6. RESOLVE EVENT
# ---------------------------------

did_succeed = (np.random.rand() < true_success_prob)

# ---------------------------------
# 7. SETTLEMENT (PAYOUTS)
# ---------------------------------

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