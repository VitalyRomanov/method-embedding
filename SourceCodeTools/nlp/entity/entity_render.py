import spacy
from SourceCodeTools.nlp import create_tokenizer

html_template = """<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>Type Prediction</title>
    <style type="text/css">
        mark {{
            background: #ddd; padding: 0.25em 0.3em; margin: 0 0.15em; line-height: 1.7; border-radius: 0.35em;
        }}
        span {{
            font-size: 0.8em; font-weight: bold; line-height: 1.7; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem
        }}
        #record
        {{
            white-space:pre-wrap;
        }}
        body {{
            font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr
        }}
    </style>
</head>

<body>
{}
</body></html>
"""


single_html_template = """<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>Source Code and Graph Alignment</title>
    <style type="text/css">
        mark {{
            background: #ddd; padding: 0.25em 0.3em; margin: 0 0.15em; line-height: 1.7; border-radius: 0.35em;
        }}
        span {{
            font-size: 0.8em; font-weight: bold; line-height: 1.7; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem
        }}
        #record
        {{
            white-space:pre-wrap;
        }}
        body {{
            font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr
        }}
    </style>
</head>

<body>
{}
</body></html>
"""

single_entry = """<div id="record">
{}
    </div>
"""

entry = """    <b>Predicted</b>
    <div id="record">
{}
    </div>
    <b>Annotated</b>
    <div id="record">
{}
    </div>
    <hr>
"""

def annotate(doc, entities):
    line = ""

    entities = sorted(entities, key=lambda x: x[0])
    # entities = list(filter(lambda x: x[2] != "Other", entities))

    for ind, t in enumerate(doc):
        if entities:
            if ind == entities[0][0]:
                line += "<mark>"
            if ind == entities[0][1]:
                line += f"<span>{entities[0][2]}</span></mark>"
                entities.pop(0)
        line += t.text + t.whitespace_

    return line


def annotate_from_spans(doc, entities):
    line = ""

    entities = sorted(entities, key=lambda x: x[0])
    # entities = list(filter(lambda x: x[2] != "Other", entities))

    for ind, c in enumerate(doc):
        if entities:
            if ind == entities[0][0]:
                line += "<mark>"
            if ind == entities[0][1]:
                line += f"<span>{entities[0][2]}</span></mark>"
                entities.pop(0)
        line += c

    return line


def render_single(text, replacements, output_path):
    # nlp = create_tokenizer("spacy")
    # doc = nlp(text)
    html = single_html_template.format(single_entry.format(annotate_from_spans(text, replacements)))
    open(output_path, "w").write(html)


def render_annotations(annotations):
    nlp = create_tokenizer("spacy")
    entries = ""
    for annotation in annotations:
        text, predicted, annotated = annotation
        doc = nlp(text[0])
        entries += entry.format(annotate(doc, predicted), annotate(doc, annotated))

    return html_template.format(entries)
