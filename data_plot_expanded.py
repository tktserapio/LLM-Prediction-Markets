import json
import matplotlib.pyplot as plt

# Load the JSON data from the file
input_file = "simulation_results_expanded.json"
with open(input_file, "r") as f:
    data = json.load(f)

# Extract rounds and market prices
rounds = [0] + [entry["round"] for entry in data]
market_prices = [0.5] + [entry["market_price"] for entry in data]

# Extract the true market price (assume it's the same for all rounds)
true_market_price = data[0]["true_success_probability"]

# Create the graph
plt.figure(figsize=(10, 6))
plt.plot(rounds, market_prices, marker="o", linestyle="-", color="b", label="Market Price")

# Add a horizontal line at the true market price
plt.axhline(y=true_market_price, color="r", linestyle="--", label=f"True Market Price (y = {true_market_price})")

# Add labels, title, and legend
plt.xlabel("Rounds")
plt.ylabel("Market Price")
plt.title("Rounds vs. Current Market Price")
plt.legend()
plt.grid(True)

output_image = "market_price_plot.png"
plt.savefig(output_image, format="png", dpi=300)
print(f"Graph saved as {output_image}")