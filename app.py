import os
import json
import re

from typing import TypedDict

from dotenv import load_dotenv
from pymongo import MongoClient

from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = "beacon_memory"
COLLECTION_NAME = "meeting_memories"

VECTOR_INDEX_NAME = "vector_index"

# Minimum similarity required for a memory to become a candidate
SIMILARITY_THRESHOLD = 0.65

# Maximum number of vector-search candidates
VECTOR_SEARCH_CANDIDATES = 5

# MiniLM embedding dimension
EMBEDDING_DIMENSIONS = 384


# ============================================================
# MONGODB CONNECTION
# ============================================================

mongo_client = MongoClient(MONGODB_URI)

db = mongo_client[DATABASE_NAME]

collection = db[COLLECTION_NAME]

print("MongoDB connected successfully! ✅")


# ============================================================
# GEMINI CONNECTION
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ============================================================
# MINILM EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("MiniLM embedding model loaded successfully! ✅")

# ============================================================
# LANGGRAPH STATE
# ============================================================

class MemoryState(TypedDict, total=False):

    meeting_summary: str

    facts: list

    new_memories: list

    relevant_memories: list

    comparisons: list

    alerts: list

    change_detected: bool


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_value(text):

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert common plural form to singular.
    #
    # This is intentionally generic and conservative.
    # It prevents simple variations such as:
    #
    # API  <-> APIs
    # deadline <-> deadlines
    # requirement <-> requirements
    #
    # from being treated as changes.
    words = text.split()

    normalized_words = []

    for word in words:

        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"

        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]

        normalized_words.append(word)

    return " ".join(normalized_words)


# ============================================================
# NODE 1
# EXTRACT IMPORTANT FACTS
# ============================================================

def extract_facts(state: MemoryState):

    print("\n================================")
    print("NODE 1: EXTRACT FACTS")
    print("================================")

    meeting_summary = state["meeting_summary"]


    prompt = f"""
You are a general-purpose long-term meeting memory extractor.

Extract important factual information from the meeting summary
that may be useful in future meetings.

The system must work for ANY type of meeting information.

Do NOT assume a fixed list of categories.

Information may include:

- decisions
- requirements
- dates
- deadlines
- people
- responsibilities
- technologies
- locations
- budgets
- quantities
- plans
- constraints
- policies
- project details
- or any other important factual information.


For every important fact, generate exactly these fields:

type
topic
value
source_text


============================================================
1. TYPE
============================================================

Choose a short, general description of what kind of information
the fact represents.

Do not use a predefined category list.

Determine the type from the meaning of the information.


============================================================
2. TOPIC
============================================================

The topic identifies the specific underlying subject.

Use a concise and meaningful topic.

If two statements refer to the same underlying subject,
try to use the same or very similar topic.

Do not create a completely different topic only because
the wording of the meeting changed.


============================================================
3. VALUE
============================================================

The value must contain ONLY the specific factual value.

The value must be the smallest meaningful piece of information
that represents the fact.

Do NOT include:

- the topic
- explanations
- surrounding context
- the complete sentence
- repeated descriptions of the topic


Examples:

Sentence:
"The application will use REST APIs to communicate with
the existing backend."

Correct:
value = "REST APIs"

Incorrect:
value = "REST API integration with the existing backend"

Incorrect:
value = "The application will use REST APIs to communicate
with the existing backend."


For a technology:
value = technology name

For a person:
value = person's name

For a date:
value = date

For a budget:
value = amount

For a location:
value = location

For a requirement:
value = concise requirement


============================================================
4. CHANGES
============================================================

If something was:

- changed
- replaced
- moved
- postponed
- increased
- decreased
- updated
- cancelled
- extended
- shortened
- switched
- modified

extract ONLY the NEW value.

Example:

"The project changed from Flutter to React Native."

Correct:
value = "React Native"

Incorrect:
value = "Flutter to React Native"

Incorrect:
value = "The project changed from Flutter to React Native."


============================================================
5. SEMANTIC VALUE CONSISTENCY
============================================================

When extracting a fact, do not repeat the topic or its context
inside the value.

Example:

Topic:
backend integration

Sentence:
"The REST API integration with the existing backend
will remain unchanged."

Correct:
value = "REST API"

Incorrect:
value = "REST API integration with the existing backend"


============================================================
6. SOURCE TEXT
============================================================

Keep the original sentence or relevant part of the meeting
summary that supports the extracted fact.


============================================================
7. AVOID DUPLICATES
============================================================

Do not extract the same fact multiple times.


============================================================
8. GENERALIZATION
============================================================

Do not create rules specifically for any particular:

- technology
- person
- project
- date
- location
- budget
- example

The extraction must work for information the system
has never seen before.


Return ONLY a valid JSON array.

Example:

[
  {{
    "type": "general information type",
    "topic": "specific underlying subject",
    "value": "current value",
    "source_text": "original supporting sentence"
  }}
]


Meeting summary:

{meeting_summary}
"""


    response = gemini_client.models.generate_content(

        model="gemini-3.5-flash-lite",

        contents=prompt
    )


    response_text = response.text.strip()


    # Remove Markdown JSON code fences if Gemini adds them
    if response_text.startswith("```"):

        response_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            response_text
        )

        response_text = re.sub(
            r"\s*```$",
            "",
            response_text
        )


    facts = json.loads(response_text)


    print("\nExtracted facts:")

    for fact in facts:

        print(
            f"- {fact['type']} | "
            f"{fact['topic']} | "
            f"{fact['value']}"
        )


    return {

        "facts": facts,

        "new_memories": []

    }


# ============================================================
# NODE 2
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(state: MemoryState):

    print("\n================================")
    print("NODE 2: CREATE EMBEDDINGS")
    print("================================")


    new_memories = []


    for fact in state["facts"]:

        text = (

            f"Type: {fact.get('type', '')}. "

            f"Topic: {fact.get('topic', '')}. "

            f"Value: {fact.get('value', '')}. "

            f"Source: {fact.get('source_text', '')}"

        )


        embedding = embedding_model.encode(text, normalize_embeddings=True).tolist()


        memory = {

            "type": fact.get("type", ""),

            "topic": fact.get("topic", ""),

            "value": fact.get("value", ""),

            "source_text": fact.get("source_text", ""),

            "text": text,

            "embedding": embedding

        }


        new_memories.append(memory)


        print(
            f"Embedding created for: "
            f"{fact.get('topic', '')}"
        )


    return {

        "new_memories": new_memories

    }


# ============================================================
# NODE 3
# SEMANTIC SEARCH
# ============================================================

def semantic_search(state: MemoryState):

    print("\n================================")
    print("NODE 3: SEARCH PREVIOUS MEMORIES")
    print("================================")


    all_relevant_memories = []


    for current_memory in state["new_memories"]:

        query_vector = current_memory["embedding"]


        results = collection.aggregate([

            {

                "$vectorSearch": {

                    "index": VECTOR_INDEX_NAME,

                    "path": "embedding",

                    "queryVector": query_vector,

                    "numCandidates": VECTOR_SEARCH_CANDIDATES,

                    "limit": VECTOR_SEARCH_CANDIDATES

                }

            },

            {

                "$project": {

                    "_id": 1,

                    "meeting_title": 1,

                    "meeting_id": 1,

                    "meeting_date": 1,

                    "type": 1,

                    "topic": 1,

                    "value": 1,

                    "source_text": 1,

                    "score": {
                        "$meta": "vectorSearchScore"
                    }

                }

            }

        ])


        candidates = list(results)


        threshold_results = []


        for memory in candidates:

            score = float(
                memory.get("score", 0)
            )


            if score >= SIMILARITY_THRESHOLD:

                threshold_results.append(memory)


        print(
            f"\nCurrent fact: "
            f"{current_memory['topic']} | "
            f"{current_memory['value']}"
        )


        print(
            f"Vector candidates above "
            f"{SIMILARITY_THRESHOLD}: "
            f"{len(threshold_results)}"
        )


        for memory in threshold_results:

            all_relevant_memories.append({

                "current_memory": current_memory,

                "previous_memory": memory

            })


            print(

                f"Candidate: "
                f"{memory.get('topic')} | "
                f"{memory.get('value')} | "
                f"Similarity: "
                f"{float(memory.get('score', 0)):.4f}"

            )


    return {

        "relevant_memories":
        all_relevant_memories

    }


# ============================================================
# NODE 4
# SAME UNDERLYING FACT CHECK
# ============================================================

def same_fact_check(state: MemoryState):

    print("\n================================")
    print("NODE 4: SAME UNDERLYING FACT CHECK")
    print("================================")


    validated_memories = []


    for item in state.get(
        "relevant_memories",
        []
    ):


        current = item["current_memory"]

        previous = item["previous_memory"]


        prompt = f"""
You are comparing two meeting memories.

Determine whether they refer to the SAME UNDERLYING FACT,
decision, requirement, property, or subject.

Do NOT decide whether the values are equal yet.

Your ONLY task is to determine whether the two memories
represent the same underlying thing.


PREVIOUS MEMORY

Type:
{previous.get('type', '')}

Topic:
{previous.get('topic', '')}

Value:
{previous.get('value', '')}

Source sentence:
{previous.get('source_text', '')}


CURRENT MEMORY

Type:
{current.get('type', '')}

Topic:
{current.get('topic', '')}

Value:
{current.get('value', '')}

Source sentence:
{current.get('source_text', '')}


Important:

- Similar wording does not automatically mean the same fact.
- Different topic wording can still represent the same fact.
- Consider the meaning of type, topic, value, and source sentence.
- Do not focus only on exact word matching.
- Do not decide whether the values are equal.
- Only decide whether they describe the same underlying fact.


Return ONLY one word:

YES

or

NO
"""


        response = gemini_client.models.generate_content(

            model="gemini-3.5-flash-lite",

            contents=prompt

        )


        answer = normalize(
            response.text
        )


        print(

            f"\nPrevious: "
            f"{previous.get('topic')} | "
            f"{previous.get('value')}"

        )


        print(

            f"Current: "
            f"{current.get('topic')} | "
            f"{current.get('value')}"

        )


        print(
            f"Same underlying fact: "
            f"{answer.upper()}"
        )


        if answer == "yes":

            validated_memories.append(item)


    print(

        f"\nValidated same-fact pairs: "
        f"{len(validated_memories)}"

    )


    return {

        "relevant_memories":
        validated_memories

    }


# ============================================================
# NODE 5
# COMPARE VALUES
# ============================================================

def compare_memories(state: MemoryState):

    print("\n================================")
    print("NODE 5: COMPARE MEMORIES")
    print("================================")


    comparisons = []

    change_detected = False


    for item in state.get(
        "relevant_memories",
        []
    ):


        current = item["current_memory"]

        previous = item["previous_memory"]


        current_value = current.get(
            "value",
            ""
        )


        previous_value = previous.get(
            "value",
            ""
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Compare normalized VALUES rather than raw strings.
        #
        # Example:
        #
        # REST API
        # REST APIs
        #
        # becomes:
        #
        # rest api
        #
        # Therefore no false change is generated.
        # ----------------------------------------------------

        normalized_current_value = normalize_value(
            current_value
        )

        normalized_previous_value = normalize_value(
            previous_value
        )


        if (
            normalized_current_value
            != normalized_previous_value
        ):


            change_detected = True


            comparison = {

                "current_type":
                current.get("type"),

                "previous_type":
                previous.get("type"),

                "current_topic":
                current.get("topic"),

                "previous_topic":
                previous.get("topic"),

                "previous_value":
                previous_value,

                "current_value":
                current_value,

                "similarity":
                previous.get("score", 0),

                "previous_meeting_id":
                previous.get("meeting_id"),

                "previous_meeting_date":
                previous.get("meeting_date")

            }


            comparisons.append(
                comparison
            )


            print("\n⚠️ CHANGE DETECTED")


            print(
                f"Previous topic: "
                f"{previous.get('topic')}"
            )


            print(
                f"Current topic: "
                f"{current.get('topic')}"
            )


            print(
                f"Previous value: "
                f"{previous_value}"
            )


            print(
                f"Current value: "
                f"{current_value}"
            )


            print(
                f"Similarity: "
                f"{float(previous.get('score', 0)):.4f}"
            )


        else:

            print("\n✅ No value change")

            print(
                f"Previous value: "
                f"{previous_value}"
            )

            print(
                f"Current value: "
                f"{current_value}"
            )


    if not change_detected:

        print("\nNo changes detected.")


    return {

        "comparisons":
        comparisons,

        "change_detected":
        change_detected

    }


# ============================================================
# NODE 6
# GENERATE ALERT
# ============================================================

def generate_alert(state: MemoryState):

    print("\n================================")
    print("NODE 6: GENERATE ALERT")
    print("================================")


    alerts = []


    for comparison in state.get(
        "comparisons",
        []
    ):


        alert = (

            f"Change detected: "

            f"{comparison['previous_value']} "

            f"→ "

            f"{comparison['current_value']}"

        )


        alerts.append(alert)


        print("\n🚨 ALERT")


        print(
            f"Previous topic: "
            f"{comparison['previous_topic']}"
        )


        print(
            f"Current topic: "
            f"{comparison['current_topic']}"
        )


        print(
            f"Previous value: "
            f"{comparison['previous_value']}"
        )


        print(
            f"Current value: "
            f"{comparison['current_value']}"
        )


        if comparison.get(
            "previous_meeting_date"
        ):

            print(
                f"Previous meeting date: "
                f"{comparison['previous_meeting_date']}"
            )


    return {

        "alerts": alerts

    }


# ============================================================
# NODE 7
# STORE CURRENT MEMORIES
# ============================================================

def store_memories(state: MemoryState):

    print("\n================================")
    print("NODE 7: STORE CURRENT MEMORIES")
    print("================================")


    stored_count = 0


    for memory in state["new_memories"]:


        memory_type = memory.get(
            "type",
            ""
        )


        topic = memory.get(
            "topic",
            ""
        )


        value = memory.get(
            "value",
            ""
        )


        existing = collection.find_one({

            "type": {

                "$regex":
                "^" + re.escape(memory_type) + "$",

                "$options":
                "i"

            },

            "topic": {

                "$regex":
                "^" + re.escape(topic) + "$",

                "$options":
                "i"

            },

            "value": {

                "$regex":
                "^" + re.escape(value) + "$",

                "$options":
                "i"

            }

        })


        if existing:

            continue


        collection.insert_one(
            memory
        )


        stored_count += 1


    if stored_count > 0:

        print(

            f"\nNew memory stored successfully! "
            f"✅ ({stored_count} new memories)"

        )

    else:

        print(
            "\nNo new memories stored."
        )


    return {}


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def check_change(
    state: MemoryState
):

    if state.get(
        "change_detected",
        False
    ):

        return "generate_alert"


    return "store_memories"


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(
    MemoryState
)


workflow.add_node(
    "extract_facts",
    extract_facts
)


workflow.add_node(
    "create_embeddings",
    create_embeddings
)


workflow.add_node(
    "semantic_search",
    semantic_search
)


workflow.add_node(
    "same_fact_check",
    same_fact_check
)


workflow.add_node(
    "compare_memories",
    compare_memories
)


workflow.add_node(
    "generate_alert",
    generate_alert
)


workflow.add_node(
    "store_memories",
    store_memories
)


# ------------------------------------------------------------
# EDGES
# ------------------------------------------------------------

workflow.add_edge(
    START,
    "extract_facts"
)


workflow.add_edge(
    "extract_facts",
    "create_embeddings"
)


workflow.add_edge(
    "create_embeddings",
    "semantic_search"
)


workflow.add_edge(
    "semantic_search",
    "same_fact_check"
)


workflow.add_edge(
    "same_fact_check",
    "compare_memories"
)


workflow.add_conditional_edges(

    "compare_memories",

    check_change,

    {

        "generate_alert":
        "generate_alert",

        "store_memories":
        "store_memories"

    }

)


workflow.add_edge(
    "generate_alert",
    "store_memories"
)


workflow.add_edge(
    "store_memories",
    END
)


# ============================================================
# COMPILE
# ============================================================

app = workflow.compile()


# ============================================================
# RUN APPLICATION
# ============================================================

print("\n")

print(
    "=============================================="
)

print(
    "       BEACON MEMORY - LANGGRAPH"
)

print(
    "=============================================="
)


print(
    f"\nSimilarity threshold: "
    f"{SIMILARITY_THRESHOLD}"
)


print(
    f"Vector search candidates: "
    f"{VECTOR_SEARCH_CANDIDATES}"
)


print(
    "\nEnter the meeting summary below."
)


print(
    "Type your complete meeting summary "
    "and press Enter.\n"
)


meeting_summary = input("> ")


result = app.invoke({

    "meeting_summary":
    meeting_summary

})


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")

print(
    "=============================================="
)

print(
    "             FINAL RESULT"
)

print(
    "=============================================="
)


if result.get(
    "change_detected"
):

    print(
        "\n⚠️ Changes detected."
    )


    for alert in result.get(
        "alerts",
        []
    ):

        print(
            f"ALERT: {alert}"
        )


else:

    print(
        "\n✅ No changes detected."
    )


print(
    "\nCurrent meeting memories "
    "have now been stored."
)


print(
    "\n=============================================="
)