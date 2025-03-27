from fastapi import FastAPI, UploadFile, File, HTTPException
import logging

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

app = FastAPI()

def run(filename: str):
    from docling.document_converter import DocumentConverter
    from pathlib import Path

    path = Path(filename)

    doc_converter = DocumentConverter()
    docs = doc_converter.convert_all([path])

    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_sdg.qa.utils import get_qa_chunks

    chunker = HierarchicalChunker()

    filters = [
        lambda chunk: len(str(chunk.text)) > 500
    ]

    dataset = {}
    for doc in docs:
        print(f"Chunking and filtering document {doc.document.name}")

        chunks = list(chunker.chunk(dl_doc=doc.document))
        qa_chunks = list(get_qa_chunks(doc.document.name, chunks, filters))
        dataset[doc.document.name] = qa_chunks[:1]

        print(f"Created dataset {doc.document.name} with {len(qa_chunks)} QA chunks")
        break

    from docling_sdg.qa.generate import Generator
    from docling_sdg.qa.base import GenerateOptions
    from docling_sdg.qa.base import LlmProviders

    generate_options = GenerateOptions(api_key="fake", project_id="project_id")
    generate_options.api_key = "fake"
    generate_options.model_id = "mixtral"
    gen = Generator(generate_options=generate_options)

    for doc, chunks in dataset.items():
        print(f"processing chunks that looks like:\n{chunks[0].text}")
        results = gen.generate_from_chunks(chunks)
        print(f"{doc}: {results.status}")
        break

    import json
    import yaml
    from textwrap import wrap

    qnas = {}
    chunk_id_to_text = {}
    with open("docling_sdg_generated_qac.jsonl", "rt") as f:
        for line in f.readlines():
            entry = json.loads(line)
            chunk_id = entry['chunk_id']
            if chunk_id not in chunk_id_to_text:
                chunk_id_to_text[chunk_id] = entry['context']
            if chunk_id not in qnas:
                qnas[chunk_id] = []
            qnas[chunk_id].append({'question': entry['question'], 'answer': entry['answer']})


    def str_presenter(dumper, data):
      if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
      elif len(data) > 80:
        data = "\n".join(wrap(data, 80))
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
      return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_presenter)

    # to use with safe_dump:
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

    class IndentedDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentedDumper, self).increase_indent(flow, False)

    data = {'seed_examples': []}
    for chunk_id, context in chunk_id_to_text.items():
        data['seed_examples'].append({
            'context': context,
            'questions_and_answers': [
                {
                    'question': example['question'],
                    'answer': example['answer'],
                } for example in qnas[chunk_id]
            ]
        })

    with open('qna.yml', 'w') as yaml_file:
        yaml.dump(data, yaml_file, Dumper=IndentedDumper, default_flow_style=False, sort_keys=False, width=80)

    print("Done")



@app.post("/generate")
async def generate(file: UploadFile = File(...)):
        try:
            contents = file.file.read()
            with open(file.filename, 'wb') as f:
                f.write(contents)
            run(file.filename)
            with open('qna.yml', 'rt') as f:
                return "\n".join(f.readlines())

        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=500, detail='Something went wrong')
        finally:
            file.file.close()

        return {"message": f"Successfully uploaded {file.filename}"}