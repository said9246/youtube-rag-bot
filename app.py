
import streamlit as st
import os
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import GenericProxyConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("🎥 YouTube Chatbot")
st.write("Ask questions about a YouTube video's content.")

video_url = st.text_input("Enter YouTube Video URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def get_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if hostname == "youtu.be":
        return path_parts[0] if path_parts else None

    if hostname == "youtube.com" or hostname.endswith(".youtube.com"):
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_video_id:
            return query_video_id

        if len(path_parts) >= 2 and path_parts[0] in (
            "embed",
            "shorts",
            "live",
        ):
            return path_parts[1]

    return None


def fetch_transcript(video_id: str):
    """Fetch a transcript, optionally routing YouTube requests through a proxy."""
    proxy_url = os.getenv("YOUTUBE_PROXY_URL", "").strip()
    proxy_config = (
        GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        if proxy_url
        else None
    )

    api = YouTubeTranscriptApi(proxy_config=proxy_config)
    return api.fetch(video_id, languages=("en", "hi", "ur"))


@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as error:
        raise RuntimeError(
            "Local embedding model could not start. Make sure PyTorch and "
            "sentence-transformers are installed correctly for your Python "
            "version. The first run also downloads the model files."
        ) from error


@st.cache_resource
def load_llm():
    try:
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.1,
        )
    except Exception as error:
        raise RuntimeError(
            "Ollama could not start. Make sure Ollama is installed, running, "
            f"and that the {OLLAMA_MODEL} model has been downloaded."
        ) from error


if st.button("Load Video"):
    video_id = get_video_id(video_url.strip())

    if not video_id:
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            with st.status("Loading video...", expanded=True) as status:
                status.write("Fetching transcript...")
                transcript = fetch_transcript(video_id)
                transcript_text = " ".join(
                    snippet.get("text", "")
                    if isinstance(snippet, dict)
                    else getattr(snippet, "text", "")
                    for snippet in transcript
                )

                if not transcript_text.strip():
                    status.update(label="Transcript is empty", state="error")
                    st.error("No transcript text found.")
                    st.stop()

                status.write("Splitting transcript into chunks...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                )
                chunks = splitter.split_text(transcript_text)

                status.write(
                    "Loading the local embedding model "
                    "(first run may take a few minutes)..."
                )
                embeddings = load_embeddings()
                status.write("Building the searchable index...")
                vector_store = FAISS.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                )

                st.session_state.vector_store = vector_store
                st.session_state.video_id = video_id
                status.update(label="Video loaded successfully", state="complete")

            st.success("Video transcript processed successfully!")
            st.write("Total chunks:", len(chunks))
            st.text_area("Transcript Preview", transcript_text[:3000], height=200)

        except (IpBlocked, RequestBlocked):
            st.error(
                "YouTube ne is app ke current IP ko block kiya hua hai. "
                "Agar aap local computer par hain to kuch der baad retry karein. "
                "Agar app Streamlit Cloud/server par chal rahi hai to residential "
                "proxy configure karna zaroori hai."
            )
            st.info(
                "Proxy set karne ke liye Windows PowerShell mein run karein:\n\n"
                '$env:YOUTUBE_PROXY_URL="http://username:password@host:port"\n\n'
                "Phir Streamlit app restart karein. Proxy URL ko public code mein "
                "hardcode na karein."
            )
        except TranscriptsDisabled:
            st.error("Is video ke owner ne captions/transcript disable kiye hue hain.")
        except NoTranscriptFound:
            st.error(
                "Is video mein English, Hindi ya Urdu transcript available nahi hai."
            )
        except VideoUnavailable:
            st.error("Video unavailable, private, age-restricted, ya delete ho chuki hai.")
        except InvalidVideoId:
            st.error("YouTube video ID invalid hai. Video URL dobara check karein.")
        except Exception as e:
            st.error(f"Could not process video: {e}")


if "vector_store" in st.session_state:
    st.subheader("Ask a Question")

    question = st.text_input("Your question about this video")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Loading local Ollama model and generating answer..."):
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 4}
                    )

                    prompt = ChatPromptTemplate.from_template(
                        """Answer the question using only the context below.g
If the answer is not in the context, say you cannot find it
in the video transcript.
    
Context:
{context}

Question: {question}
Answer:"""
                    )

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {
                            "context": retriever | format_docs,
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | load_llm()
                        | StrOutputParser()
                    )

                    answer = rag_chain.invoke(question)

                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.error(
                    f"Answer generation failed: {e}. "
                    f"Check that Ollama is running and {OLLAMA_MODEL} is installed."
                )











