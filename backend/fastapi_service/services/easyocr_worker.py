import sys
import json
import logging

try:
    import torch
    import easyocr
except Exception as e:
    sys.stdout.write(json.dumps({"error": f"Failed to import ML libs: {e}"}) + "\n")
    sys.exit(1)

def main():
    gpu = torch.cuda.is_available()
    try:
        reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    except Exception as e:
        sys.stdout.write(json.dumps({"error": f"Failed to init reader: {e}"}) + "\n")
        sys.exit(1)

    sys.stdout.write(json.dumps({"status": "ready", "gpu": gpu}) + "\n")
    sys.stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
        except:
            continue
        
        if req.get("action") == "quit":
            break
        
        path = req.get("image_path")
        if not path:
            print(json.dumps({"error": "No image_path provided"}))
            sys.stdout.flush()
            continue
            
        try:
            res = reader.readtext(path, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7)
            # Serialize
            out = []
            for bbox, text, conf in res:
                # convert float32 to float, int32 to int for JSON
                out.append({
                    "bbox": [[float(x), float(y)] for x, y in bbox],
                    "text": str(text),
                    "confidence": float(conf)
                })
            sys.stdout.write(json.dumps({"results": out}) + "\n")
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            
        sys.stdout.flush()

if __name__ == "__main__":
    main()
