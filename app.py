import os

import re

import json

import hashlib

from pathlib import Path

from urllib.parse import urlparse, parse_qs
import textwrap



import streamlit as st

from dotenv import load_dotenv



from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq





# ============================================================

# 1. PAGE CONFIGURATION AND STYLING

# ============================================================



st.set_page_config(

    page_title="VidQuery — AI Video Knowledge Assistant",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="collapsed",

)



load_dotenv()



DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_DIR = Path("faiss_indexes")

INDEX_DIR.mkdir(exist_ok=True)



st.markdown(

    """

    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');



    html, body, [class*="css"] {

        font-family: 'DM Sans', sans-serif;

    }

    h1, h2, h3 {

        font-family: 'Manrope', sans-serif;

        letter-spacing: -0.035em;

    }

    .block-container {

        max-width: 1050px;

        padding-top: 2.2rem;

        padding-bottom: 4rem;

    }

    .hero {

        padding: 1.5rem 1.6rem;

        border: 1px solid rgba(128,128,128,.22);

        border-radius: 20px;

        background: linear-gradient(120deg, rgba(124,58,237,.13), rgba(59,130,246,.08));

        margin-bottom: 1.3rem;

    }

    .hero-brand {
        display: flex;
        align-items: center;
        gap: .65rem;
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: -.025em;
        margin-bottom: 1.1rem;
    }
    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: .65rem;
        color: white;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        font-size: 1rem;
        font-weight: 800;
    }
    .hero-kicker {

        text-transform: uppercase;

        letter-spacing: .13em;

        font-size: .72rem;

        font-weight: 700;

        opacity: .65;

        margin-bottom: .55rem;

    }

    .hero-title {

        font-family: 'Manrope', sans-serif;

        font-size: clamp(1.8rem, 4vw, 2.6rem);

        font-weight: 800;

        line-height: 1.1;

        margin: 0;

    }

    .hero-copy {

        opacity: .72;

        margin-top: .6rem;

        max-width: 650px;

        font-size: .98rem;

    }

    .video-card {

        border: 1px solid rgba(128,128,128,.22);

        border-radius: 16px;

        padding: 1rem 1.1rem;

        margin: .7rem 0 1rem 0;

    }

    .muted {

        opacity: .65;

        font-size: .88rem;

    }

    .source-card {

        border: 1px solid rgba(128,128,128,.18);

        border-radius: 12px;

        padding: .75rem .9rem;

        margin: .45rem 0;

    }

    div[data-testid="stChatMessage"] {

        border-radius: 14px;

    }

    div.stButton > button {

        border-radius: 10px;

        font-weight: 600;

    }

    div[data-testid="stTextInput"] input {

        border-radius: 10px;

    }

    </style>

    """,

    unsafe_allow_html=True,

)





# ============================================================

# 2. CACHED MODEL LOADERS

# ============================================================



@st.cache_resource(show_spinner="Loading the local embedding model...")

def load_embeddings(model_name: str):

    return HuggingFaceEmbeddings(model_name=model_name)





@st.cache_resource(show_spinner=False)

def load_llm(model_name: str, temperature: float):

    return ChatGroq(

        model=model_name,

        temperature=temperature,

    )





# ============================================================

# 3. YOUTUBE TRANSCRIPT AND CHUNKING

# ============================================================



def extract_video_id(value: str) -> str:

    """Accept a video ID or common YouTube URL formats."""

    value = value.strip()



    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):

        return value



    parsed = urlparse(value if "://" in value else f"https://{value}")

    host = parsed.netloc.lower().replace("www.", "")



    if host in {"youtu.be", "m.youtu.be"}:

        candidate = parsed.path.strip("/").split("/")[0]

    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:

        if parsed.path == "/watch":

            candidate = parse_qs(parsed.query).get("v", [""])[0]

        else:

            parts = [part for part in parsed.path.split("/") if part]

            candidate = parts[1] if len(parts) >= 2 and parts[0] in {

                "embed", "shorts", "live"

            } else ""

    else:

        candidate = ""



    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate or ""):

        return candidate



    raise ValueError("Paste a valid YouTube video URL or an 11-character video ID.")





def extract_transcript(video_id: str):

    """Fetch English captions and retain text-position to video-time mappings."""

    api = YouTubeTranscriptApi()

    fetched = api.fetch(video_id, languages=["en"])



    text_parts = []

    timestamp_offsets = []

    snippet_rows = []

    current_offset = 0



    for snippet in fetched:

        text = snippet.text.strip()

        if not text:

            continue



        timestamp_offsets.append((current_offset, float(snippet.start)))

        text_parts.append(text)

        snippet_rows.append(

            {

                "start_seconds": float(snippet.start),

                "duration_seconds": float(snippet.duration),

                "text": text,

            }

        )

        current_offset += len(text) + 1



    transcript = " ".join(text_parts)

    if not transcript:

        raise ValueError("No transcript text was returned for this video.")



    return transcript, timestamp_offsets, snippet_rows





def timestamp_for_position(position: int, timestamp_offsets):

    timestamp = 0.0

    for offset, seconds in timestamp_offsets:

        if offset <= position:

            timestamp = seconds

        else:

            break

    return timestamp





def create_chunks(

    transcript: str,

    timestamp_offsets,

    video_id: str,

    chunk_size: int,

    chunk_overlap: int,

):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,

        separators=["\n\n", "\n", ". ", " ", ""],

    )



    split_texts = splitter.split_text(transcript)

    documents = []

    search_position = 0



    for chunk_number, chunk in enumerate(split_texts):

        position = transcript.find(chunk, search_position)

        if position == -1:

            position = transcript.find(chunk)

        if position == -1:

            position = 0



        start_time = timestamp_for_position(position, timestamp_offsets)

        source_url = (

            f"https://www.youtube.com/watch?v={video_id}"

            f"&t={int(start_time)}s"

        )



        documents.append(

            Document(

                page_content=chunk,

                metadata={

                    "video_id": video_id,

                    "chunk_id": chunk_number,

                    "start_time": start_time,

                    "source": source_url,

                },

            )

        )



        search_position = max(

            position + 1,

            position + len(chunk) - chunk_overlap,

        )



    return documents





# ============================================================

# 4. FAISS INDEX CACHING

# ============================================================



def build_or_load_vector_store(

    documents,

    video_id: str,

    transcript: str,

    embeddings,

    embedding_model: str,

    chunk_size: int,

    chunk_overlap: int,

):

    """

    Reuse the local FAISS index only if transcript and all index-relevant

    configuration values match.

    """

    transcript_hash = hashlib.sha256(

        transcript.encode("utf-8")

    ).hexdigest()



    config = {

        "video_id": video_id,

        "transcript_hash": transcript_hash,

        "embedding_model": embedding_model,

        "chunk_size": chunk_size,

        "chunk_overlap": chunk_overlap,

    }



    config_hash = hashlib.sha256(

        json.dumps(config, sort_keys=True).encode("utf-8")

    ).hexdigest()



    safe_video_id = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)

    index_path = INDEX_DIR / f"{safe_video_id}_{config_hash[:12]}"

    manifest_path = index_path / "manifest.json"



    index_exists = (

        (index_path / "index.faiss").exists()

        and (index_path / "index.pkl").exists()

        and manifest_path.exists()

    )



    if index_exists:

        try:

            manifest = json.loads(

                manifest_path.read_text(encoding="utf-8")

            )

            if manifest.get("config_hash") == config_hash:

                # Only load indexes created and stored locally by this app.

                store = FAISS.load_local(

                    str(index_path),

                    embeddings,

                    allow_dangerous_deserialization=True,

                )

                return store, "Reused your saved index", config_hash, True

        except Exception:

            # A damaged or incompatible cache is rebuilt below.

            pass



    store = FAISS.from_documents(documents, embeddings)

    index_path.mkdir(parents=True, exist_ok=True)

    store.save_local(str(index_path))



    manifest = {

        "config_hash": config_hash,

        **config,

        "vector_count": store.index.ntotal,

        "embedding_dimensions": store.index.d,

    }

    manifest_path.write_text(

        json.dumps(manifest, indent=2),

        encoding="utf-8",

    )



    return store, "Created and saved a new index", config_hash, False





# ============================================================

# 5. CONVERSATION, RETRIEVAL, AND ANSWERING

# ============================================================



def format_history(history, last_n=4):

    recent = history[-last_n:]

    return "\n".join(

        f"User: {item['question']}\nAssistant: {item['answer']}"

        for item in recent

    )





def rewrite_question(question: str, history, llm):

    if not history:

        return question



    prompt = PromptTemplate(

        template=(

            "Rewrite the latest question as a standalone question using "

            "the conversation history. Resolve references and pronouns. "

            "Preserve the user's meaning. If it is already standalone, "

            "return it unchanged. Return only the rewritten question.\n\n"

            "Conversation:\n{history}\n\n"

            "Latest question: {question}\n\nStandalone question:"

        ),

        input_variables=["history", "question"],

    )



    return (prompt | llm | StrOutputParser()).invoke(

        {

            "history": format_history(history),

            "question": question,

        }

    ).strip()





def format_documents(documents):

    sections = []

    for doc in documents:

        seconds = int(doc.metadata.get("start_time", 0))

        sections.append(

            f"Timestamp: {seconds}s\n"

            f"Content:\n{doc.page_content}"

        )

    return "\n\n---\n\n".join(sections)





def run_rag(

    question: str,

    vector_store,

    history,

    llm,

    embeddings,

    top_k: int,

    fetch_k: int,

    lambda_mult: float,

):

    standalone_question = rewrite_question(question, history, llm)

    query_vector = embeddings.embed_query(standalone_question)



    candidate_count = min(fetch_k, vector_store.index.ntotal)

    result_count = min(top_k, vector_store.index.ntotal)



    similarity_results = vector_store.similarity_search_with_relevance_scores(

        standalone_question,

        k=candidate_count,

    )



    retrieved_docs = vector_store.max_marginal_relevance_search(

        standalone_question,

        k=result_count,

        fetch_k=candidate_count,

        lambda_mult=lambda_mult,

    )



    context = format_documents(retrieved_docs)

    history_text = format_history(history)



    prompt = PromptTemplate(

        template="""You are a helpful assistant for a YouTube video.



Answer the user's question using ONLY the transcript context below.

Do not use outside knowledge or invent facts.

If the transcript does not contain enough information, say:

"I couldn't find that in this video."

Be clear, useful, and concise. When useful, mention the video timestamp.



Conversation history:

{history}



Transcript context:

{context}



Question:

{question}



Answer:""",

        input_variables=["history", "context", "question"],

    )



    prompt_value = prompt.invoke(

        {

            "history": history_text,

            "context": context,

            "question": standalone_question,

        }

    )



    answer = (prompt | llm | StrOutputParser()).invoke(

        {

            "history": history_text,

            "context": context,

            "question": standalone_question,

        }

    )



    return {

        "answer": answer,

        "sources": [

            {

                "chunk_id": doc.metadata.get("chunk_id"),

                "start_time": float(doc.metadata.get("start_time", 0)),

                "url": doc.metadata.get("source", ""),

                "text": doc.page_content,

            }

            for doc in retrieved_docs

        ],

        "trace": {

            "original_question": question,

            "standalone_question": standalone_question,

            "query_vector": query_vector,

            "similarity_results": similarity_results,

            "retrieved_docs": retrieved_docs,

            "context": context,

            "prompt": prompt_value.to_string(),

        },

    }





# ============================================================

# 6. OPTIONAL VIDEO SUMMARY

# ============================================================



def summarize_video(documents, llm, progress_callback=None):

    map_prompt = PromptTemplate(

        template="""Summarize this transcript section. Capture the main ideas,

important facts, explanations and examples. Do not add outside information.



Transcript section:

{text}



Section summary:""",

        input_variables=["text"],

    )



    reduce_prompt = PromptTemplate(

        template="""Combine these partial summaries into one coherent video

summary. Remove repetition, preserve important concepts and do not invent facts.



Partial summaries:

{text}



Complete video summary:""",

        input_variables=["text"],

    )



    map_chain = map_prompt | llm | StrOutputParser()

    reduce_chain = reduce_prompt | llm | StrOutputParser()



    partial_summaries = []

    total = len(documents)



    for index, document in enumerate(documents):

        partial_summaries.append(

            map_chain.invoke({"text": document.page_content})

        )

        if progress_callback:

            progress_callback((index + 1) / max(total, 1))



    if not partial_summaries:

        return "No transcript chunks were available to summarize."



    while len(partial_summaries) > 1:

        reduced = []

        for start in range(0, len(partial_summaries), 8):

            batch = partial_summaries[start:start + 8]

            reduced.append(

                reduce_chain.invoke({"text": "\n\n".join(batch)})

            )

        partial_summaries = reduced



    return partial_summaries[0]





# ============================================================

# 7. SESSION STATE

# ============================================================



DEFAULT_STATE = {

    "video_id": None,

    "transcript": None,

    "timestamp_offsets": [],

    "snippet_rows": [],

    "documents": [],

    "vector_store": None,

    "index_message": None,

    "config_hash": None,

    "cache_hit": False,

    "chat_history": [],

    "video_summary": None,

    "last_trace": None,

}



for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value





# ============================================================

# 8. SIDEBAR — SETTINGS, NOT THE MAIN EXPERIENCE

# ============================================================



with st.sidebar:

    st.markdown("## VidQuery")
    st.caption("AI video knowledge assistant")
    st.divider()
    st.markdown("### Settings")



    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    with st.expander("Advanced pipeline settings"):

        llm_model = st.text_input(

            "Groq model",

            value=DEFAULT_LLM_MODEL,

        )

        embedding_model = st.text_input(

            "Embedding model",

            value=DEFAULT_EMBEDDING_MODEL,

        )

        chunk_size = st.slider(

            "Chunk size",

            min_value=200,

            max_value=1600,

            value=800,

            step=100,

        )

        chunk_overlap = st.slider(

            "Chunk overlap",

            min_value=0,

            max_value=300,

            value=120,

            step=20,

        )

        top_k = st.slider(

            "Retrieved chunks",

            min_value=1,

            max_value=10,

            value=5,

        )

        fetch_k = st.slider(

            "MMR candidate pool",

            min_value=5,

            max_value=30,

            value=15,

        )

        fetch_k = max(fetch_k, top_k)

        lambda_mult = st.slider(

            "MMR relevance/diversity",

            min_value=0.0,

            max_value=1.0,

            value=0.7,

            step=0.05,

        )

        temperature = st.slider(

            "LLM temperature",

            min_value=0.0,

            max_value=1.0,

            value=0.0,

            step=0.1,

        )



    st.caption(

        "The embedding model runs locally. Groq is used to rewrite questions "

        "when needed and generate answers."

    )



    if st.button("Clear chat", use_container_width=True):

        st.session_state.chat_history = []

        st.session_state.last_trace = None

        st.rerun()



    if st.button("Reset video", use_container_width=True):

        for key, value in DEFAULT_STATE.items():

            st.session_state[key] = value

        st.rerun()





# ============================================================

# 9. MAIN USER INTERFACE

# ============================================================



st.markdown(
    textwrap.dedent("""
        <div class="hero">
            <div class="hero-brand">
                <span class="brand-mark">V</span>
                <span>VidQuery</span>
            </div>
            <div class="hero-kicker">AI-powered video knowledge assistant</div>
            <div class="hero-title">Don't just watch. Understand.</div>
            <div class="hero-copy">
                Turn YouTube videos into searchable knowledge. Ask questions in plain
                English, explore key ideas, and jump directly to the relevant moments.
            </div>
        </div>
    """),
    unsafe_allow_html=True,
)



with st.form("video_form", clear_on_submit=False):

    video_input = st.text_input(

        "YouTube video link",

        placeholder="https://www.youtube.com/watch?v=...",

        label_visibility="visible",

    )

    process_col, info_col = st.columns([1, 2])

    with process_col:

        process_clicked = st.form_submit_button(

            "Process video",

            type="primary",

            use_container_width=True,

        )

    with info_col:

        st.markdown(

            '<div class="muted" style="padding-top:.55rem">'

            "English captions must be available for this video."

            "</div>",

            unsafe_allow_html=True,

        )





# Process the video. A Groq key is NOT required for transcript extraction,

# chunking, embedding, or FAISS index creation.

if process_clicked:

    try:

        video_id = extract_video_id(video_input)



        with st.status("Getting your video ready...", expanded=True) as status:

            st.write("Reading the YouTube transcript...")

            transcript, offsets, snippets = extract_transcript(video_id)



            st.write("Preparing transcript sections...")

            documents = create_chunks(

                transcript,

                offsets,

                video_id,

                chunk_size,

                chunk_overlap,

            )



            st.write("Loading the embedding model...")

            embedding_object = load_embeddings(embedding_model)



            st.write("Checking your saved search index...")

            vector_store, index_message, config_hash, cache_hit = (

                build_or_load_vector_store(

                    documents,

                    video_id,

                    transcript,

                    embedding_object,

                    embedding_model,

                    chunk_size,

                    chunk_overlap,

                )

            )



            st.session_state.video_id = video_id

            st.session_state.transcript = transcript

            st.session_state.timestamp_offsets = offsets

            st.session_state.snippet_rows = snippets

            st.session_state.documents = documents

            st.session_state.vector_store = vector_store

            st.session_state.index_message = index_message

            st.session_state.config_hash = config_hash

            st.session_state.cache_hit = cache_hit

            st.session_state.chat_history = []

            st.session_state.video_summary = None

            st.session_state.last_trace = None



            status.update(

                label="Your video is ready!",

                state="complete",

                expanded=False,

            )



        st.success("Video processed successfully. You can start asking questions.")



    except TranscriptsDisabled:

        st.error("Captions are disabled for this video.")

    except Exception as error:

        st.error(f"Couldn't process this video: {error}")





# ============================================================

# 10. READY STATE — VIDEO, CHAT, AND SOURCES

# ============================================================



if st.session_state.video_id and st.session_state.vector_store is not None:

    video_id = st.session_state.video_id

    documents = st.session_state.documents

    vector_store = st.session_state.vector_store



    st.markdown('<div class="video-card">', unsafe_allow_html=True)

    video_col, details_col = st.columns([1.25, 1])



    with video_col:

        st.video(f"https://www.youtube.com/watch?v={video_id}")



    with details_col:

        st.markdown("### Video ready")

        st.markdown(

            f'<div class="muted">Video ID: <code>{video_id}</code></div>',

            unsafe_allow_html=True,

        )

        st.write(

            "Ask questions about the explanations, concepts, examples, "

            "or key takeaways from this video."

        )



        metric1, metric2 = st.columns(2)

        metric1.metric("Transcript chunks", len(documents))

        metric2.metric(

            "Video length",

            f"{int(st.session_state.snippet_rows[-1]['start_seconds'] // 60)}+ min"

            if st.session_state.snippet_rows else "—",

        )



        if st.session_state.cache_hit:

            st.caption("✓ Saved index reused — no need to rebuild embeddings.")

        else:

            st.caption("✓ Search index created and saved locally.")



    st.markdown("</div>", unsafe_allow_html=True)



    st.markdown("### Ask about this video")

    st.caption("Ask a question below. You can also ask follow-up questions.")



    # Render the conversation first so chat_input remains at the bottom.

    for message in st.session_state.chat_history:

        with st.chat_message("user"):

            st.markdown(message["question"])



        with st.chat_message("assistant"):

            st.markdown(message["answer"])



            if message.get("sources"):

                with st.expander("Sources from this video"):

                    for source_number, source in enumerate(message["sources"], 1):

                        seconds = int(source["start_time"])

                        minutes, remaining_seconds = divmod(seconds, 60)

                        st.markdown(

                            f"**{source_number}. [{minutes:02d}:{remaining_seconds:02d}]"

                            f"({source['url']})**"

                        )

                        preview = source["text"]

                        if len(preview) > 420:

                            preview = preview[:420].rstrip() + "…"

                        st.caption(preview)



    user_question = st.chat_input(

        "Ask VidQuery about this video...",

        key="video_chat_input",

    )



    if user_question:

        if not groq_key:

            st.warning(

                "Add your Groq API key in Settings (top-left sidebar) to ask questions."

            )

        else:

            try:

                llm = load_llm(llm_model, temperature)

                embedding_object = load_embeddings(embedding_model)



                with st.spinner("Finding the relevant parts of the video..."):

                    result = run_rag(

                        question=user_question,

                        vector_store=vector_store,

                        history=st.session_state.chat_history,

                        llm=llm,

                        embeddings=embedding_object,

                        top_k=top_k,

                        fetch_k=fetch_k,

                        lambda_mult=lambda_mult,

                    )



                st.session_state.chat_history.append(

                    {

                        "question": user_question,

                        "answer": result["answer"],

                        "sources": result["sources"],

                        "trace": result["trace"],

                    }

                )

                st.session_state.last_trace = result["trace"]

                st.rerun()



            except Exception as error:

                st.error(f"Couldn't generate an answer: {error}")



    # Optional summary: useful, but not part of the default chat workflow.

    with st.expander("Summarize the entire video"):

        st.caption(

            "Creates a summary from all transcript chunks. This can take longer "

            "and uses multiple LLM calls for longer videos."

        )



        if st.button("Generate video summary", type="secondary"):

            if not groq_key:

                st.warning("Add your Groq API key in Settings first.")

            else:

                try:

                    llm = load_llm(llm_model, temperature)

                    progress = st.progress(0.0, text="Summarizing transcript...")

                    summary = summarize_video(

                        documents,

                        llm,

                        progress_callback=lambda value: progress.progress(

                            value,

                            text=f"Summarizing transcript: {value:.0%}",

                        ),

                    )

                    st.session_state.video_summary = summary

                    progress.empty()

                    st.rerun()

                except Exception as error:

                    st.error(f"Summary generation failed: {error}")



        if st.session_state.video_summary:

            st.markdown(st.session_state.video_summary)



    # ========================================================

    # 11. OPTIONAL DEVELOPER INSIGHTS

    # ========================================================



    with st.expander("Developer insights · Inspect the RAG pipeline"):

        st.caption(

            "These technical details are hidden during normal use. Open this "

            "section when you want to inspect how retrieval and generation work."

        )



        trace = st.session_state.last_trace



        st.markdown("#### Processing overview")

        overview1, overview2, overview3 = st.columns(3)

        overview1.metric("Transcript snippets", len(st.session_state.snippet_rows))

        overview2.metric("Chunks", len(documents))

        overview3.metric("FAISS vectors", vector_store.index.ntotal)



        st.caption(

            f"FAISS dimensions: {vector_store.index.d} · "

            f"Index status: {st.session_state.index_message}"

        )



        with st.expander("Transcript and chunking"):

            st.write(

                f"Transcript length: {len(st.session_state.transcript):,} characters"

            )

            st.write(

                f"Chunk size: {chunk_size} · Overlap: {chunk_overlap}"

            )

            st.text_area(

                "Transcript preview",

                st.session_state.transcript[:5000],

                height=180,

                disabled=True,

            )



            selected_chunk = st.selectbox(

                "Inspect a chunk",

                options=list(range(len(documents))),

                format_func=lambda number: f"Chunk {number + 1}",

            )

            selected_document = documents[selected_chunk]

            st.code(selected_document.page_content, language="text")

            st.json(selected_document.metadata)



        with st.expander("FAISS index and vector mapping"):

            st.write("Index type:", type(vector_store.index).__name__)

            st.write("Number of vectors:", vector_store.index.ntotal)

            st.write("Dimensions:", vector_store.index.d)



            mapping_rows = []

            for vector_id, document_id in vector_store.index_to_docstore_id.items():

                document = vector_store.docstore.search(document_id)

                mapping_rows.append(

                    {

                        "Vector ID": vector_id,

                        "Document ID": document_id,

                        "Chunk ID": document.metadata.get("chunk_id"),

                        "Start time": document.metadata.get("start_time"),

                        "Text preview": document.page_content[:150],

                    }

                )

            st.dataframe(

                mapping_rows,

                use_container_width=True,

                hide_index=True,

            )



            st.caption(

                "The local cache is reused only when the transcript hash, "

                "embedding model, chunk size, and overlap match."

            )



        if trace:

            with st.expander("Latest query and rewritten question"):

                st.write("Original question")

                st.info(trace["original_question"])

                st.write("Standalone question sent to retrieval")

                st.success(trace["standalone_question"])



            with st.expander("Query embedding"):

                query_vector = trace["query_vector"]

                st.write(f"Dimensions: {len(query_vector)}")

                st.code(

                    json.dumps(

                        [round(float(value), 6) for value in query_vector[:30]],

                        indent=2,

                    ),

                    language="json",

                )



            with st.expander("Similarity search candidates"):

                candidate_rows = []

                for rank, (document, score) in enumerate(

                    trace["similarity_results"], 1

                ):

                    candidate_rows.append(

                        {

                            "Rank": rank,

                            "Relevance score": round(float(score), 4),

                            "Chunk ID": document.metadata.get("chunk_id"),

                            "Timestamp": document.metadata.get("start_time"),

                            "Text": document.page_content[:220],

                        }

                    )

                st.dataframe(

                    candidate_rows,

                    use_container_width=True,

                    hide_index=True,

                )

                st.caption(

                    "These are similarity-search candidates. MMR then selects "

                    "a relevant and diverse subset."

                )



            with st.expander("Final MMR-selected chunks"):

                for rank, document in enumerate(trace["retrieved_docs"], 1):

                    seconds = int(document.metadata.get("start_time", 0))

                    minutes, remaining_seconds = divmod(seconds, 60)

                    st.markdown(

                        f"**Chunk {rank} · {minutes:02d}:{remaining_seconds:02d}**"

                    )

                    st.code(document.page_content, language="text")



            with st.expander("Context passed to the LLM"):

                st.code(trace["context"], language="text")



            with st.expander("Full prompt sent to Groq"):

                st.code(trace["prompt"], language="text")



else:

    st.markdown(
    textwrap.dedent("""
        <div style="text-align:center; padding:2.2rem 1rem 1rem 1rem;">
            <div style="font-size:2.2rem;">🎥</div>
            <h3 style="margin:.4rem 0;">Ready when you are</h3>
            <div class="muted">
                Add a YouTube link above and select <b>Process video</b>.
                Your chat will appear here once the transcript is ready.
            </div>
        </div>
    """),
    unsafe_allow_html=True,
)



st.divider()

st.markdown(

    '<div class="muted" style="text-align:center">'

    "VidQuery · Don't just watch. Understand."

    "</div>",

    unsafe_allow_html=True,

)
