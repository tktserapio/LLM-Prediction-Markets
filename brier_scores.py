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

# -------------------------
# 2) OPENAI HELPER
# -------------------------

def chat_completion(system_text, user_text):
    """
    Replace this with your actual GPT-4 or Azure call as needed.
    Right now, just a stub referencing the 'openai.chat.completions.create' call.
    """
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.0,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
    )

    # AZURE AI
    # client = ChatCompletionsClient(
    #     endpoint="https://models.github.ai/inference",
    #     credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    # )

    # response = client.complete(
    #     messages=[
    #         SystemMessage(system_text),
    #         UserMessage(user_text),
    #     ],
    #     model="openai/gpt-4.1-mini",
    #     temperature=1.0,
    #     max_tokens=4096,
    #     top_p=0.1
    # )


    return response.choices[0].message.content

# -------------------------
# 3) SUBAGENT ESTIMATION FUNCTIONS
# -------------------------
# We define multiple versions of weather/traffic bots, plus new bots, each
# returning a float in [0,1].

def weatherbot_estimate(bot_name, true_is_rainy):
    """
    This version can have varied accuracy depending on the bot_name.
    For example, WeatherBot1 might be more accurate than WeatherBot2.
    We'll do this by changing how likely the bot is to see the correct weather.
    """
    # Suppose WeatherBot1 is 90% accurate at identifying if it's rainy,
    # while WeatherBot2 is only 70% accurate, etc.
    if "WeatherBot1" in bot_name:
        accuracy = 0.90
    elif "WeatherBot2" in bot_name:
        accuracy = 0.80
    else:
        # default fallback
        accuracy = 0.80

    sees_rainy_correctly = (np.random.rand() < accuracy)
    # If it's actually rainy & sees it: "rainy"
    # If it's not rainy & sees it: "clear"
    # If sees incorrectly: invert
    if true_is_rainy:
        weather_context = "rainy" if sees_rainy_correctly else "clear"
    else:
        weather_context = "clear" if sees_rainy_correctly else "rainy"

    system_text = """
    You are WeatherBot. 
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route A to deliver a package will arrive on time.
    Your only basis is partial knowledge about the weather, which is described below.
    """
    user_text = f"""
    You are {bot_name}. The weather you observe is: '{weather_context}'.
    Estimate the probability of success of taking Route A.
    Return only a float between 0.0 and 1.0.
    """

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        print(f"Error parsing response: {text}")
        return 0.5

def trafficbot_estimate(bot_name, true_is_heavy_traffic):
    """
    Similarly, multiple TrafficBots with different reliability.
    """
    if "TrafficBot1" in bot_name:
        traffic_accuracy = 0.85
    elif "TrafficBot2" in bot_name:
        traffic_accuracy = 0.70
    else:
        traffic_accuracy = 0.75

    sees_traffic_correctly = (np.random.rand() < traffic_accuracy)
    if true_is_heavy_traffic:
        traffic_context = "heavy traffic" if sees_traffic_correctly else "light traffic"
    else:
        traffic_context = "light traffic" if sees_traffic_correctly else "heavy traffic"

    system_text = """
    You are TrafficBot. 
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route A to deliver a package will arrive on time.
    Your only basis is partial knowledge about the traffic, which is described below.
    """
    user_text = f"""You see: '{traffic_context}'. 
    Estimate the probability of success of taking Route A.
    Return only a float between 0.0 and 1.0."""

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def maintenancebot_estimate():
    # same as before, random chance of seeing "potholes" or "smooth"
    route_condition = np.random.choice(["potholes", "smooth"])
    system_text = """You are a MaintenanceBot. You see partial info about the road condition. 
    Output a probability in [0,1] for successful on-time delivery.
    """
    user_text = f"""The route condition is: {route_condition}. 
    Estimate the probability of success of taking Route A.
    Return only a float between 0.0 and 1.0."""
    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def local_events_bot_estimate(bot_name):
    """
    Another new type of subagent that sees local events (like protests/festivals).
    We'll randomize the 'observed' event, which may or may not match reality.
    """
    # Suppose LocalEventsBot is fairly uncertain. Let it see a random pick from these events:
    possible_events = ["protest near route", "big festival downtown", "normal day", "sports event nearby"]
    observed_event = np.random.choice(possible_events)
    
    system_text = """
    You are LocalEventsBot.
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route A to deliver a package will arrive on time.
    Your only basis is partial knowledge about local events, which is described below.
    """
    user_text = f"""You see: {observed_event}. 
    Estimate the probability of success of taking Route A.
    Return only a float between 0.0 and 1.0."""

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def road_condition_bot_estimate(bot_name):
    """
    A more fine-grained version of MaintenanceBot.
    """
    possible_conditions = ["severe potholes", "minor cracks", "freshly paved", "wet and slippery"]
    observed_condition = np.random.choice(possible_conditions)
    system_text = """
    You are RoadConditionBot.
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route A to deliver a package will arrive on time.
    Your only basis is partial knowledge about the road conditions, which is described below.
    """
    user_text = f"""Observed condition: {observed_condition}. 
    Estimate the probability of success of taking Route A.
    Return only a float between 0.0 and 1.0."""
    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

# -------------------------
# 5) DECIDE_TRADE
# -------------------------
def decide_trade(subagent_name, p_est, market_price, capital):
    system_prompt = (
        "You are a simulated agent in an internal prediction market about whether Route A will deliver on time. "
        "You have a personal probability estimate, the current market price, and your available capital. "
        "You can buy shares (positive number) if you think the event is underpriced, "
        "or short shares (negative number) if you think it's overpriced. A share pays out 1 if the event occurs, 0 otherwise."
        "Your TOP PRIORITY is to maximize your profit in the long run."
    )

    user_text = (
        f"Agent: {subagent_name}\n"
        f"Your probability estimate: {p_est:.3f}\n"
        f"Market price: {market_price:.3f}\n"
        f"Capital: {capital:.2f}\n"
        "Decide how many shares to buy (positive) or short (negative). Provide a brief justification that should be within 100 words. End with a single float number."
    )

    response_text = chat_completion(system_prompt, user_text)
    
    # parse out numeric value
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

def simulate_one():
    # 1) TRUE WORLD STATE
    is_rainy         = np.random.rand() < 0.30
    is_heavy_traffic = np.random.rand() < 0.40
    true_prob = (0.30 if is_rainy and is_heavy_traffic else
                 0.60 if is_rainy else
                 0.50 if is_heavy_traffic else
                 0.95)
    # extra factors
    if np.random.rand() < 0.05: true_prob *= 0.80
    if np.random.rand() < 0.10: true_prob *= 0.85
    true_success_prob = min(max(true_prob, 0.0), 1.0)

    # 2) Build subagents
    subagent_specs = [
        ("WeatherBot1",     lambda: weatherbot_estimate("WeatherBot1",     is_rainy)),
        ("WeatherBot2",     lambda: weatherbot_estimate("WeatherBot2",     is_rainy)),
        ("TrafficBot1",     lambda: trafficbot_estimate("TrafficBot1",     is_heavy_traffic)),
        ("TrafficBot2",     lambda: trafficbot_estimate("TrafficBot2",     is_heavy_traffic)),
        ("MaintBot",        maintenancebot_estimate),
        ("LocalEventsBot1", lambda: local_events_bot_estimate("LocalEventsBot1")),
        ("RoadConditionBot1", lambda: road_condition_bot_estimate("RoadConditionBot1")),
    ]
    subagent_estimates   = {name: fn() for name, fn in subagent_specs}
    subagent_capital     = {name:100.0 for name in subagent_estimates}
    subagent_positions   = {name:  0.0 for name in subagent_estimates}

    market_price = 0.50
    liquidity    = 1/8
    num_rounds   = 10

    # 3) Trading
    for _ in range(num_rounds):
        for name, p_est in subagent_estimates.items():
            capital    = subagent_capital[name]
            trade_size = decide_trade(name, p_est, market_price, capital)

            if trade_size > 0:
                cost = trade_size * market_price
                if cost > capital:
                    trade_size = float(int(capital // market_price))
                    cost = trade_size * market_price
                subagent_capital[name]   -= cost
                subagent_positions[name] += trade_size
                diff = abs(p_est - market_price)
                market_price += diff * liquidity * (trade_size / 100.0)

            elif trade_size < 0:
                shares_to_short = abs(trade_size)
                if shares_to_short > subagent_capital[name]:
                    shares_to_short = int(subagent_capital[name])
                    trade_size = -float(shares_to_short)
                credit = shares_to_short * market_price
                subagent_capital[name] += credit
                if subagent_capital[name] < 0:
                    subagent_capital[name] -= credit
                    shares_to_short = 0
                    trade_size      = 0
                if shares_to_short > 0:
                    subagent_positions[name] -= shares_to_short
                    diff = abs(p_est - market_price)
                    market_price -= diff * liquidity * (shares_to_short / 100.0)

        market_price = max(0.0, min(1.0, market_price))

    # 4) Outcome & Settlement
    did_succeed = (np.random.rand() < true_success_prob)
    for name, pos in subagent_positions.items():
        if pos > 0 and did_succeed:
            subagent_capital[name] += pos
        elif pos < 0 and did_succeed:
            subagent_capital[name] -= abs(pos)

    return market_price, did_succeed

# -------------------------
# 7) MULTI-RUN & BRIER SCORE
# -------------------------

N     = 20
preds = []
obs   = []

for i in range(N):
    p, success = simulate_one()
    preds.append(p)
    obs.append(1 if success else 0)

preds = np.array(preds)
obs   = np.array(obs)
brier = np.mean((preds - obs) ** 2)

print(f"Ran {N} sims. Brier score={brier:.4f}")

# Optionally save for further analysis
np.savez("brier_data.npz", preds=preds, obs=obs)