from typing import Union


def make_html_start(title: Union[str, None] = None):

    style = """    
        * {
          box-sizing: border-box;
        }

        body {
            margin-left: 2rem;
        }
        header {
          padding-top: 0.5rem;
          height: 5rem;
        }
        .sample {
            font-family: sans-serif;
            font-weight: 500;
            font-size: 1.2rem;
            width: max(60vw, 60rem);
            border-bottom: 2px solid #aaa;
            padding: 0.5rem 0 0.5rem 0;
          }
        .audio-wrapper {
            display: flex;               
            align-items: center;
            justify-content: space-between;
            width: 60rem;
            flex-wrap: wrap;
        }
        .audio-wrapper label {
            display: inline-block;         
            width: 3.5rem;
        }
        .audio-row {
            display: flex;
            align-items: center;
        }
        audio {
            height: 2rem;
            width: 22rem;
            margin-right: 1rem;
        }
        .text-arabic {
            font-size: 1.6rem;
            margin: 0.5rem;
         }
        .row-title {
            width: 6rem;
         }
    """

    title = f"<title>{title}</title>" if title is not None else ""
    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Samples</title>
        <style>
        {style}
        </style>
        {title}
    </head>
    <body>
    """
    return html


def make_html_end():
    html = """</body>
    </html>
    """
    return html


def make_sample_entry(wav_path: str, text: str):
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    html = f"""<div class="sample">
      <audio src="{wav_path}" controls></audio>
      <br />
      {text}
    </div>
    """
    return html


def make_sample_entry2(wav_path: str, text0: str,
                       text1: str, ar_dir: str = 'ltr'):
    text0 = text0.replace('<', '&lt;').replace('>', '&gt;')
    text1 = text1.replace('<', '&lt;').replace('>', '&gt;')
    html = f"""<div class="sample">
      <audio src="{wav_path}" controls></audio>
      <div class="text-arabic" dir="{ar_dir}">
        {text0}
      </div>      
      {text1}
    </div>
    """
    return html


def make_double_entry(wav_gen: str, wav_ref: str,
                      text0: str, text1: str, ar_dir: str = 'ltr'):
    text0 = text0.replace('<', '&lt;').replace('>', '&gt;')
    text1 = text1.replace('<', '&lt;').replace('>', '&gt;')
    html = f"""<div class="sample">
      <div class="audio-wrapper">
        <label>Generated:</label>
        <audio src="{wav_gen}" controls></audio>
        <label>Reference:</label>
        <audio src="{wav_ref}" controls></audio>
      </div>
      <div class="text-arabic" dir="{ar_dir}">
        {text0}
      </div>      
      {text1}
    </div>
    """
    return html

def make_multi_entry(wavs_list, row_titles,
                      text0: str, text1: str, ar_dir: str='ltr'):
    text0 = text0.replace('<', '&lt;').replace('>', '&gt;')
    text1 = text1.replace('<', '&lt;').replace('>', '&gt;')

    rows = ""
    for i in range(0,len(wavs_list),2):     
      row_title = row_titles[i//2] 
      rows += f"""<div class="audio-row">
          <span class="row-title">{row_title}</span>
          <label>{wavs_list[i][0]}:</label>
          <audio src="{wavs_list[i][1]}" controls></audio>    
          <label>{wavs_list[i+1][0]}:</label>
          <audio src="{wavs_list[i+1][1]}" controls></audio> 
      </div>  
      """

    html = f"""<div class="sample">
      <div class="audio-wrapper">
        {rows} 
      </div>
      <div class="text-arabic" dir="{ar_dir}">
        {text0}
      </div>      
      {text1}
    </div>
    """


    return html

def make_h_tag(text: str, n: int = 2):
    html = f"""<h{n}>{text}</h{n}>
    """
    return html


def make_img_tag(src: str, alt: str = ""):
    html = f"""<img src="{src}" alt="{alt}" />
    """
    return html


def make_volume_script(volume: float = 0.35):
    html = f"""<script>
      const audioOutputs = document.querySelectorAll('audio');
      audioOutputs.forEach(a => a.volume = {volume});
    </script>
    """
    return html
