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

ROUTES        = ["A", "B", "C"]
liquidity     = 1/8
num_rounds    = 10        # feel free to raise back to 10
ROUTE_MULT    = {"A":1.00, "B":0.85, "C":1.10}   # used later

def weatherbot_estimate(bot_name, route, true_is_rainy):
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

    system_text = f"""
    You are WeatherBot. 
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route {route} to deliver a package will arrive on time.
    Your only basis is partial knowledge about the weather, which is described below.
    """
    user_text = f"""
    You are {bot_name}. The weather you observe is: '{weather_context}'.
    Estimate the probability of success of taking Route {route}.
    Return only a float between 0.0 and 1.0.
    """

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        print(f"Error parsing response: {text}")
        return 0.5

def trafficbot_estimate(bot_name, route, true_is_heavy_traffic):
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

    system_text = f"""
    You are TrafficBot. 
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route {route} to deliver a package will arrive on time.
    Your only basis is partial knowledge about the traffic, which is described below.
    """
    user_text = f"""You see: '{traffic_context}'. 
    Estimate the probability of success of taking Route {route}.
    Return only a float between 0.0 and 1.0."""

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def maintenancebot_estimate(route):
    # same as before, random chance of seeing "potholes" or "smooth"
    route_condition = np.random.choice(["potholes", "smooth"])
    system_text = f"""You are a MaintenanceBot. You see partial info about the road condition. 
    Output a probability in [0,1] for successful on-time delivery.
    """
    user_text = f"""The route condition is: {route_condition}. 
    Estimate the probability of success of taking Route {route}.
    Return only a float between 0.0 and 1.0."""
    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def local_events_bot_estimate(bot_name, route):
    """
    Another new type of subagent that sees local events (like protests/festivals).
    We'll randomize the 'observed' event, which may or may not match reality.
    """
    # Suppose LocalEventsBot is fairly uncertain. Let it see a random pick from these events:
    possible_events = ["protest near route", "big festival downtown", "normal day", "sports event nearby"]
    observed_event = np.random.choice(possible_events)
    
    system_text = f"""
    You are LocalEventsBot.
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route {route} to deliver a package will arrive on time.
    Your only basis is partial knowledge about local events, which is described below.
    """
    user_text = f"""You see: {observed_event}. 
    Estimate the probability of success of taking Route {route}.
    Return only a float between 0.0 and 1.0."""

    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5

def road_condition_bot_estimate(bot_name, route):
    """
    A more fine-grained version of MaintenanceBot.
    """
    possible_conditions = ["severe potholes", "minor cracks", "freshly paved", "wet and slippery"]
    observed_condition = np.random.choice(possible_conditions)
    system_text = f"""
    You are RoadConditionBot.
    You are part of a coordinated team of specialized AI assistants (subagents). 
    Your role is to contribute a reliable numeric probability between 0.0 and 1.0 for whether taking Route {route} to deliver a package will arrive on time.
    Your only basis is partial knowledge about the road conditions, which is described below.
    """
    user_text = f"""Observed condition: {observed_condition}. 
    Estimate the probability of success of taking Route {route}.
    Return only a float between 0.0 and 1.0."""
    text = chat_completion(system_text, user_text)
    try:
        return float(text.strip())
    except ValueError:
        return 0.5


def decide_trade(subagent_name, route, p_est, market_price_scalar, capital):
    system_prompt = (
        f"You are a simulated agent in an internal prediction market about whether taking Route {route} will deliver on time."
        "You can buy shares (positive number) if you think the event is underpriced, "
        "or short shares (negative number) if you think it's overpriced. A share pays out 1 if the event occurs, 0 otherwise."
        "Your TOP PRIORITY is to maximize your profit in the long run."
    )

    user_text = (
        f"Agent: {subagent_name}\n"
        f"Your probability estimate: {p_est:.3f}\n"
        f"Market price: {market_price_scalar:.3f}\n"
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

    return trade_size, response_text

def simulate_one():
    # 1) TRUE WORLD STATE
    is_rainy, is_heavy = np.random.rand()<0.30, np.random.rand()<0.40
    base_prob = 0.30 if (is_rainy and is_heavy) else 0.60 if is_rainy else 0.50 if is_heavy else 0.95
    if np.random.rand()<0.05: base_prob *= 0.80
    if np.random.rand()<0.10: base_prob *= 0.85
    true_success_prob = {r: min(max(base_prob*ROUTE_MULT[r],0),1) for r in ROUTES}

    # 2) Build subagents
    subagent_specs = [
        ("WeatherBot1",     lambda r: weatherbot_estimate("WeatherBot1", r, is_rainy)),
        ("WeatherBot2",     lambda r: weatherbot_estimate("WeatherBot2", r, is_rainy)),
        ("TrafficBot1",     lambda r: trafficbot_estimate("TrafficBot1", r, is_heavy)),
        ("TrafficBot2",     lambda r: trafficbot_estimate("TrafficBot2", r, is_heavy)),
        ("MaintBot",        lambda r: maintenancebot_estimate(r)),
        ("LocalEventsBot1", lambda r: local_events_bot_estimate("LocalEventsBot1", r)),
        ("RoadConditionBot1", lambda r: road_condition_bot_estimate("RoadConditionBot1", r)),
    ]
    subagent_estimates = {r:{} for r in ROUTES}
    for agent, fn in subagent_specs:
        for r in ROUTES:
            subagent_estimates[r][agent] = fn(r)
    subagent_capital   = {agent:100.0 for agent,_ in subagent_specs}
    subagent_positions = {r:{agent:0.0 for agent,_ in subagent_specs} for r in ROUTES}
    market_price       = {r:0.5 for r in ROUTES}

    trade_history = []

    # 3) Trading
    for rnd in range(1, num_rounds+1):
        for r in ROUTES:
            for agent in subagent_estimates[r]:
                p_est = subagent_estimates[r][agent]
                cap   = subagent_capital[agent]
                p0    = market_price[r]

                trade, full_resp = decide_trade(agent, r, p_est, p0, cap)

                entry = dict(round=rnd, route=r, agent=agent, belief=p_est,
                             price_before=p0, trade_size=trade,
                             capital_before=cap, true_success_prob=true_success_prob[r],
                             full_response=full_resp)

                if trade>0:
                    cost = trade*p0
                    if cost>cap:
                        trade = int(cap//p0); cost = trade*p0
                    subagent_capital[agent] -= cost
                    subagent_positions[r][agent] += trade
                    market_price[r] += abs(p_est-p0)*liquidity*(trade/100)
                elif trade<0:
                    qty = abs(trade)
                    if qty>cap:
                        qty=int(cap); trade=-qty
                    credit = qty*p0
                    subagent_capital[agent] += credit
                    subagent_positions[r][agent] -= qty
                    market_price[r] -= abs(p_est-p0)*liquidity*(qty/100)

                market_price[r] = max(0.0,min(1.0,market_price[r]))
                entry.update(price_after=market_price[r],
                             capital_after=subagent_capital[agent],
                             position=subagent_positions[r][agent])
                trade_history.append(entry)

    # 4) Outcome & Settlement
    did_succeed = {r:(random.random()<true_success_prob[r]) for r in ROUTES}
    for r in ROUTES:
        for agent,q in subagent_positions[r].items():
            if q>0 and did_succeed[r]:
                subagent_capital[agent]+=q
            elif q<0 and did_succeed[r]:
                subagent_capital[agent]-=abs(q)

    return dict(market_price=market_price, did_succeed=did_succeed), trade_history

# -------------------------
# 7) MULTI-RUN & BRIER SCORE
# -------------------------

N, preds, obs, all_runs = 20, [], [], []
for run in range(1,N+1):
    summary, hist = simulate_one()
    for e in hist:
        e["run"]=run; e["success"]=summary["did_succeed"][e["route"]]
    all_runs.extend(hist)
    for route,price in summary["market_price"].items():
        preds.append(price)
        obs.append(1 if summary["did_succeed"][route] else 0)

brier = np.mean((np.array(preds)-np.array(obs))**2)

with open("prediction_market_log.json","w") as f:
    json.dump(all_runs,f,indent=2)
np.savez("brier_data.npz",preds=np.array(preds),obs=np.array(obs))

print(f"Ran {N} sims across {len(ROUTES)} routes → {len(preds)} forecasts.")
print(f"Brier score = {brier:.4f}")