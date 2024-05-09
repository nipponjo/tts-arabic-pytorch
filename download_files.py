# %%
import pathlib
import gdown

# %%

FILES_DICT = {  
    # TACOTRON  
    "tacotron2_ar_mse.pth": {
        "path": "pretrained/tacotron2_ar_mse.pth",
        "url": "https://drive.google.com/file/d/1GCu-ZAcfJuT5qfzlKItcNqtuVNa7CNy9/view?usp=sharing",
        "download": True,
    },
    "tacotron2_ar_adv.pth": {
        "path": "pretrained/tacotron2_ar_adv.pth",
        "url": "https://drive.google.com/file/d/1FusCFZIXSVCQ9Q6PLb91GIkEnhn_zWRS/view?usp=sharing",
        "download": True,
    },
    # FASTPITCH
    "fastpitch_ar_mse.pth": {
        "path": "pretrained/fastpitch_ar_mse.pth",
        "url": "https://drive.google.com/file/d/1sliRc62wjPTnPWBVQ95NDUgnCSH5E8M0/view?usp=sharing",
        "download": True,
    },
    "fastpitch_ar_adv.pth": {
        "path": "pretrained/fastpitch_ar_adv.pth",
        "url": "https://drive.google.com/file/d/1-vZOhi9To_78-yRslC6sFLJBUjwgJT-D/view?usp=sharing",
        "download": True,
    },
    "fastpitch_ar_ms.pth": {
        "path": "pretrained/fastpitch_ar_ms.pth",
        "url": "https://drive.google.com/file/d/18IYUSRXvLErVjaDORj_TKzUxs90l61Ja/view?usp=sharing",
        "download": True,
    },
    # HIFIGAN
    "hifigan-asc.pth": {
        "path": "pretrained/hifigan-asc-v1/hifigan-asc.pth",
        "url": "https://drive.google.com/file/d/1zSYYnJFS-gQox-IeI71hVY-fdPysxuFK/view?usp=sharing",
        "download": True,
    },
    # DIACRITIZERS
    "shakkelha_rnn_3_big_20.pth": {
        "path": "pretrained/diacritizers/shakkelha_rnn_3_big_20.pth",
        "url": "https://drive.google.com/file/d/1CbDjbuBr-798x88vjLGtMPSB2Y1KwD68/view?usp=sharing",
        "download": True,
    },
    "shakkala_second_model6.pth": {
        "path": "pretrained/diacritizers/shakkala_second_model6.pth",
        "url": "https://drive.google.com/file/d/1hgMGqXLTc58Gq_bN7WpuBWscBxX-rXXd/view?usp=sharing",
        "download": True,
    },    
    
}

# %%

root_dir = pathlib.Path(__file__).parent

for file_dict in FILES_DICT.values():
    file_path = root_dir.joinpath(file_dict['path'])

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
    if file_path.exists():
        print(file_dict['path'], "already exists!")
    elif file_dict.get('download', True):
        print("Downloading ", file_dict['path'], "...")
        output_filepath = gdown.download(file_dict['url'], output=file_path.as_posix(), fuzzy=True)
