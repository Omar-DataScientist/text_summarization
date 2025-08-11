from transformers import pipeline

# Step 1: Load model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 2: Long text
article = """
Renewable energy is becoming increasingly important as the world confronts the dual challenges of climate change and depleting fossil fuel resources. Unlike fossil fuels such as coal, oil, and natural gas, renewable energy sources—including solar, wind, hydro, and geothermal—are naturally replenished and have a much smaller environmental footprint. Over the past decade, technological advancements have significantly reduced the cost of renewable energy systems, making them more competitive with traditional energy sources. Solar panels have become more efficient, wind turbines have grown larger and more productive, and energy storage solutions, such as lithium-ion batteries, have improved to store excess energy for use when production is low. Governments across the globe are introducing policies and incentives to encourage the adoption of renewable technologies, including tax credits, subsidies, and renewable energy mandates. However, challenges remain, particularly in integrating these variable energy sources into existing power grids. Fluctuations in energy production require flexible grid management and backup power systems, often still reliant on fossil fuels. Furthermore, large-scale renewable projects can have environmental and social impacts, such as habitat disruption and land-use conflicts. Public awareness and support for renewable energy continue to grow, but widespread adoption will require ongoing innovation, investment, and political will. The transition to renewable energy is not just an environmental necessity; it also offers economic opportunities, from creating jobs in the green technology sector to reducing energy import dependence. As the world moves toward a more sustainable future, renewable energy will play a pivotal role in shaping a cleaner, healthier planet.

"""

# Step 3: Summarize
summary = summarizer(article, max_length=130, min_length=30, do_sample=False)

# Step 4: Output
print("Original Text:\n", article)
print("\nGenerated Summary:\n", summary[0]['summary_text'])
