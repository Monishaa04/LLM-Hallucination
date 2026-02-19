import matplotlib.pyplot as plt
import textwrap

# Style and font tuning for neat, professional-looking labels
plt.style.use("ggplot")
plt.rcParams.update({
	"font.family": "sans-serif",
	"font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
	"axes.titlesize": 16,
	"axes.labelsize": 12,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
})

models = [
	"RAG (Ours)",
	"LLaMA-3.1-8B Instant",
	"OpenAI GPT-OSS-20B",
	"MoonshotAI Kimi-K2-0905",
	"Meta LLaMA-4 Maverick-17B",
	"Qwen Qwen-3-32B",
]


weighted_accuracy = [0.451, 0.539, 0.569, 0.502, 0.578, 0.539]
hallucination_rate = [0.0, 0.275, 0.157, 0.333, 0.255, 0.235]
abstention_rate = [0.216, 0.0, 0.0, 0.0, 0.0, 0.0]


def tidy_label(name, width=14):
	return textwrap.fill(name, width=width)


def bar_plot(title, values, ylabel):
	x = list(range(len(models)))
	labels = [tidy_label(m, width=14) for m in models]
	fig, ax = plt.subplots(figsize=(10, 5))
	bars = ax.bar(x, values, color="#4C72B0", edgecolor="#2A3F5F")
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	ax.set_ylim(0, 1)
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=0, ha="center")
	ax.grid(axis="y", linestyle="--", alpha=0.6)
	for bar, val in zip(bars, values):
		ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
	plt.subplots_adjust(bottom=0.28)
	plt.tight_layout()
	plt.show()


bar_plot("Weighted Accuracy Comparison", weighted_accuracy, "Weighted Accuracy")
bar_plot("Hallucination Rate Comparison", hallucination_rate, "Hallucination Rate")
bar_plot("Abstention Rate Comparison", abstention_rate, "Abstention Rate")
