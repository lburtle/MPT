import json

with open("/output/signal.json", "r") as f:
    bnn_data = json.load(f)

with open("/output/result.json", "r") as f:
    sentiment_data = json.load(f)

avg_sentiment = sentiment_data.get("avg_sentiment")
ovr_sentiment = sentiment_data.get("overall_sentiment")

pred_close = bnn_data.get("predicted_close")
signal = bnn_data.get("signal")

score = (avg_sentiment + signal) / 3  ## Force rudimentary concensus between BNN and sentiment

if score > 0.7:
    output = "Invest"
else:
    output = "Do not invest"

print(output)