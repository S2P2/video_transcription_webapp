import gradio as gr
import os
import ffmpeg
import time
from torch import cuda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import CharacterTextSplitter
from faster_whisper import WhisperModel, BatchedInferencePipeline
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

langfuse_handler = CallbackHandler(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host="http://localhost:3000"
)

WHISPER_CT_MODEL_NAME = "terasut/whisper-th-large-v3-combined-ct2"
# WHISPER_CT_MODEL_NAME = "model/moonsoon-whisper-medium-gigaspeech2-ct2"
# model_name = "scb10x/monsoon-whisper-medium-gigaspeech2"

LLM_MODEL_NAME = "gemma2:27b-instruct-q8_0"
# LLM_MODEL_NAME = "gemma2:9b-instruct-q8_0"
# LLM_MODEL_NAME = "llama3.1:70b"
# LLM_MODEL_NAME = "gemma2:9b",
# LLM_MODEL_NAME = "qwen2.5:14b-instruct-q8_0",

device = "cuda" if cuda.is_available() else "cpu"

model = WhisperModel(WHISPER_CT_MODEL_NAME, device=device, compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

def transcribe_audio(audio_filepath):

    start_time = time.time()

    segments, info = batched_model.transcribe(audio_filepath, batch_size=16)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output = []
    output_with_timestamps = []
    
    for segment in segments:
        formatted_string = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(formatted_string)
        output.append(segment.text)
        output_with_timestamps.append(formatted_string)

    output = "\n".join(output)
    output_with_timestamps = "\n".join(output_with_timestamps)

    run_time = time.time() - start_time
    
    with open("./tmp/output.txt", "w") as text_file:
        text_file.write(output)
    
    with open("./tmp/output_with_timestamps.txt", "w") as text_file:
        text_file.write(output_with_timestamps)

    return output, output_with_timestamps, run_time

def transcribe_video(video_filepath, video_start_time, video_end_time):
    start_time = time.time()

    # create tmp directory for intermediate audio file
    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    if video_start_time == 0 and video_end_time == 0:
        out, _ = (ffmpeg
        .input(video_filepath)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )
    else:
        out, _ = (ffmpeg
        .input(video_filepath, ss=video_start_time, t=video_end_time-video_start_time)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )

    output, output_with_timestamps, audio_run_time = transcribe_audio('./tmp/output.wav')

    run_time = time.time() - start_time

    return output, output_with_timestamps, run_time

def copy_from_text_to_text(text):
    return text

async def post_process_text(input_text):

    start_time = time.time()
    
    import operator
    from typing import Annotated, List, Literal, TypedDict

    from langchain.chains.combine_documents.reduce import (
        acollapse_docs,
        split_list_of_docs,
    )
    from langchain_core.documents import Document
    from langgraph.constants import Send
    from langgraph.graph import END, START, StateGraph

    map_prompt = ChatPromptTemplate.from_messages(
        # [("system", "Write a concise summary of the following (in original language):\\n\\n{context}")]
        [("system", "สรุปการสนทนาต่อไปนี้ แยกเป็นหัวข้อ:\n\n{context}")]
)
    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0,
        num_ctx=8192,
        # other params...
    )    

    map_chain = map_prompt | llm | StrOutputParser()

    reduce_template = """
        โปรดใช้ภาษาไทยเป็นหลัก (ปนภาษาอังกฤษได้) ถึงแม้ว่าคำสั่งจะเป็นภาษาอังกฤษ
        The following is a set of summaries from meeting transcript:
        {docs}
        Take these and combined them into one detailed final summary.
        you can use Thai and English, but your user are Thai
        """

    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

    reduce_chain = reduce_prompt | llm | StrOutputParser()


    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=1000, chunk_overlap=0
    # )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=6000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    split_docs = [Document(page_content=x) for x in text_splitter.split_text(input_text)]
    print(f"Generated {len(split_docs)} documents.")

    token_max = 10000

    def length_function(documents: List[Document]) -> int:
        # """Get number of tokens for input contents."""
        # for doc in documents:
        #     print(llm.get_num_tokens(doc.page_content))
        # return sum(llm.get_num_tokens(doc.page_content) for doc in documents)
        """Get number of characterss for input contents."""
        return sum(len(doc.page_content) for doc in documents)


    # This will be the overall state of the main graph.
    # It will contain the input document contents, corresponding
    # summaries, and a final summary.
    class OverallState(TypedDict):
        # Notice here we use the operator.add
        # This is because we want combine all the summaries we generate
        # from individual nodes back into one list - this is essentially
        # the "reduce" part
        contents: List[str]
        summaries: Annotated[list, operator.add]
        collapsed_summaries: List[Document]
        final_summary: str


    # This will be the state of the node that we will "map" all
    # documents to in order to generate summaries
    class SummaryState(TypedDict):
        content: str


    # Here we generate a summary, given a document
    async def generate_summary(state: SummaryState):
        response = await map_chain.ainvoke(state["content"])
        return {"summaries": [response]}


    # Here we define the logic to map out over the documents
    # We will use this an edge in the graph
    def map_summaries(state: OverallState):
        # We will return a list of `Send` objects
        # Each `Send` object consists of the name of a node in the graph
        # as well as the state to send to that node
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]


    def collect_summaries(state: OverallState):
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }


    # Add node to collapse summaries
    async def collapse_summaries(state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], length_function, token_max
        )
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

        return {"collapsed_summaries": results}


    # This represents a conditional edge in the graph that determines
    # if we should collapse the summaries or not
    def should_collapse(
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = length_function(state["collapsed_summaries"])
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"


    # Here we will generate the final summary
    async def generate_final_summary(state: OverallState):
        response = await reduce_chain.ainvoke(state["collapsed_summaries"])
        return {"final_summary": response}


    # Construct the graph
    # Nodes:
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)  # same as before
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges:
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()

    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        config={"recursion_limit": 10, "callbacks": [langfuse_handler]}
    ):
        print(list(step.keys()))

    # chain = prompt | llm | StrOutputParser()

    # output_text = chain.invoke({"text": "".join(chunks)})

    with open("./tmp/summary.txt", "w") as text_file:
        text_file.write(step['generate_final_summary']['final_summary'])

    run_time = time.time() - start_time

    return step['generate_final_summary']['final_summary'], run_time, step

with gr.Blocks() as demo:
    gr.Markdown("# Video/Audio Transcription app")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video()
            video_start_time = gr.Number(label="Start Time (in seconds)")
            video_end_time = gr.Number(label="End Time (in seconds)\n0 for both Start and End Time for extract all")
            transcribe_video_button = gr.Button("Transcribe from video")
        with gr.Column():
            audio_input = gr.Audio(type="filepath")
            transcribe_audio_button = gr.Button("Transcribe from audio")
    transcribe_run_time = gr.Number(label="Total run time : (in seconds)", precision=2)
    with gr.Row():
        with gr.Column():
            text_output = gr.TextArea(label="Transcript", show_copy_button=True, interactive=False)
        with gr.Column():
            text_output_timestamps = gr.TextArea(label="Transcript with timestamps", show_copy_button=True, interactive=False)

    gr.Markdown("# Summarization")
    copy_from_output_button = gr.Button("Copy from transcript")
    post_process_text_input = gr.TextArea(label="Text input for post-process", show_copy_button=True)
    post_process_button = gr.Button("Summarize text")
    # post_process_text_output = gr.TextArea(label="Text output for post-process", show_copy_button=True)
    post_process_run_time = gr.Number(label="Total run time : (in seconds)", precision=2)
    post_process_markdown_output = gr.Markdown(label="Text output for post-process", show_copy_button=True)

    transcribe_video_button.click(transcribe_video, inputs=[video_input, video_start_time, video_end_time], outputs=[text_output, text_output_timestamps, transcribe_run_time])
    transcribe_audio_button.click(transcribe_audio, inputs=audio_input, outputs=[text_output, text_output_timestamps, transcribe_run_time])
    copy_from_output_button.click(copy_from_text_to_text, inputs=text_output, outputs = post_process_text_input)
    post_process_button.click(post_process_text, inputs=post_process_text_input, outputs=[post_process_markdown_output, post_process_run_time])

if __name__ == "__main__":
    demo.launch()