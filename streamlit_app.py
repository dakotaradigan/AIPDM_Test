import os
import json
from typing import List, Dict, Tuple

import openai
import streamlit as st

# ------------------ Configuration ------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "Benchmark_Min_Correlation_AI.json"
CORRELATION_THRESHOLD = 0.9
WEIGHT_TOLERANCE = 2  # percent

# ------------------ Helper Functions ------------------

def load_benchmark_data(path: str = DATA_PATH) -> Dict[str, dict]:
    """Load benchmark data from JSON file and return dict keyed by benchmark name."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {item["benchmark"]: item for item in raw}
    except FileNotFoundError:
        st.error("Benchmark data file not found. Please add Benchmark_Min_Correlation_AI.json to the project.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Benchmark JSON is malformed. Please correct the file and restart the app.")
        st.stop()


def parse_blend_via_openai(prompt: str, valid_benchmarks: List[str]) -> Tuple[List[Dict[str, float]], str]:
    """Use OpenAI to parse user prompt into list of dicts with benchmark and weight.
    Returns (blend, raw_tool_response).
    """
    system_msg = (
        "You are a helpful assistant that extracts asset allocation mixes."
        "\nReturn ONLY JSON in the form: {\"blend\": [{\"benchmark\": <name>, \"weight\": <number>} , ...]}"
        "\nBenchmarks MUST be chosen from this list (case sensitive): "
        + ", ".join(valid_benchmarks)
        + ". Do not include any other keys or commentary."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0
        )
        assistant_content = response.choices[0].message.content.strip()
        blend_data = json.loads(assistant_content)["blend"]
        return blend_data, assistant_content
    except Exception as e:
        # Fallback to basic heuristic parsing if OpenAI fails
        blend = []
        try:
            # Very naive: look for patterns like "60% S&P 500" or "S&P 500 60%"
            import re
            percent_pattern = re.compile(r"(\d{1,3})%\s*([A-Za-z0-9 &]+)")
            for pct, name in percent_pattern.findall(prompt):
                name = name.strip()
                if name in valid_benchmarks:
                    blend.append({"benchmark": name, "weight": float(pct)})
            # Alternative order
            alt_pattern = re.compile(r"([A-Za-z0-9 &]+)\s*(\d{1,3})%")
            for name, pct in alt_pattern.findall(prompt):
                name = name.strip()
                if name in valid_benchmarks and all(d["benchmark"] != name for d in blend):
                    blend.append({"benchmark": name, "weight": float(pct)})
            return blend, "heuristic_fallback"
        except Exception:
            return [], str(e)


def validate_blend(blend: List[Dict[str, float]], benchmark_data: Dict[str, dict]) -> Tuple[bool, List[str]]:
    """Validate benchmarks exist and weights are reasonable. Returns (is_valid, error_messages)."""
    errors = []
    if not blend:
        errors.append("Unable to parse any benchmarks from the input.")
        return False, errors

    total_weight = sum(item.get("weight", 0) for item in blend)
    if abs(total_weight - 100) > WEIGHT_TOLERANCE:
        errors.append(f"Weights should sum to 100. Current total is {total_weight}.")

    for item in blend:
        name = item.get("benchmark")
        if name not in benchmark_data:
            errors.append(f"Benchmark '{name}' is not supported.")

    return len(errors) == 0, errors


def apply_minimum_suppression(min_list: List[Tuple[str, float, float]], data: Dict[str, dict]) -> Tuple[List[Tuple[str, float, float]], bool]:
    """If any pair of benchmarks has high correlation, set both their minimums to the higher of the two.
    Returns (updated_list, suppression_applied)"""
    updated = min_list.copy()
    suppression_applied = False
    for i, (bench_i, w_i, min_i) in enumerate(updated):
        for j in range(i + 1, len(updated)):
            bench_j, w_j, min_j = updated[j]
            corr = data[bench_i]["correlations"].get(bench_j) or data[bench_j]["correlations"].get(bench_i) or 0
            if corr >= CORRELATION_THRESHOLD:
                higher = max(min_i, min_j)
                if higher != min_i or higher != min_j:
                    suppression_applied = True
                updated[i] = (bench_i, w_i, higher)
                updated[j] = (bench_j, w_j, higher)
    return updated, suppression_applied


def calculate_blend_minimum(blend: List[Dict[str, float]], data: Dict[str, dict]) -> Tuple[float, bool]:
    """Compute weighted or adjusted minimum based on correlations. Returns (minimum, suppression_used)"""
    # Prepare list of tuples (benchmark, weight, minimum)
    min_list = [(item["benchmark"], item["weight"], data[item["benchmark"]]["minimum"]) for item in blend]

    # Apply suppression for high correlations
    min_list, suppression = apply_minimum_suppression(min_list, data)

    total = sum(weight / 100 * minimum for _, weight, minimum in min_list)
    return total, suppression


def assess_confidence(is_valid: bool, errors: List[str]) -> str:
    if not is_valid:
        return "low"
    if errors:
        return "medium"
    return "high"


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def generate_rationale(confidence: str, had_errors: bool, suppressed: bool) -> str:
    if confidence == "high":
        base = "All benchmarks recognized with clear weights and correlations."
    elif confidence == "medium":
        base = "We parsed the blend but found minor issues such as weight rounding or unclear correlations."
    else:
        return "We encountered issues parsing your request. Please review benchmarks and weights provided."

    if suppressed:
        base += " High correlations detected – applied minimum suppression to avoid double-counting risk."
    return base

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="AUM Minimum Chatbot", page_icon="💬", layout="centered")
st.title("AUM Minimum Advisor Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Each message: {role, content, response, confidence}

# Load data once
benchmark_data = load_benchmark_data()
valid_benchmarks = list(benchmark_data.keys())

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Show action buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Copy Response", key=f"copy_{idx}"):
                    st.experimental_clipboard(msg["content"])
                    st.toast("Response copied to clipboard")
            with col2:
                if msg.get("confidence", "high") == "low":
                    if st.button("Escalate to Product", key=f"esc_{idx}"):
                        st.toast("Your query has been escalated.")

# Prompt for user input
user_input = st.chat_input("Ask about AUM minimums …")
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process input
    blend, raw = parse_blend_via_openai(user_input, valid_benchmarks)
    is_valid, validation_errors = validate_blend(blend, benchmark_data)

    if is_valid:
        blend_minimum, suppressed = calculate_blend_minimum(blend, benchmark_data)
        confidence = assess_confidence(is_valid, validation_errors)
        rationale = generate_rationale(confidence, bool(validation_errors), suppressed)

        response_text = (
            f"**AUM Minimum: {format_currency(blend_minimum)}**\n\n"
            f"{rationale}\n\n"
            f"Confidence: **{confidence.title()}**"
        )
    else:
        confidence = "low"
        error_lines = "\n".join(f"- {e}" for e in validation_errors)
        response_text = (
            f"We couldn't process your request due to the following issues:\n{error_lines}\n\n"
            "Please try again or escalate to product support."
        )

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_text, "confidence": confidence})

    # Rerun to display updated conversation immediately
    st.experimental_rerun()