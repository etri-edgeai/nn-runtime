import marshal
import os
from pathlib import Path, PurePosixPath
from glob import glob
import argparse
import logging

logging.basicConfig(    
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)

def obfuscate(
        target_folder:str, 
        output_folder:str="dist"
    ):

    result = []
    for j in ["*.py"]:
        result.extend([y for x in os.walk(target_folder) for y in glob(os.path.join(x[0], j))])

    try:
        for i in result:
            logging.info(f"{i} file obfuscate")
            with open(i, "rb") as f:
                code = compile(f.read(), "", mode='exec', dont_inherit=True)
                py_bytes = marshal.dumps(code)
                a = str(py_bytes)
            tmp_path = PurePosixPath(i)
            tmp_path = tmp_path.relative_to(target_folder)
            target_path = os.path.join(output_folder, tmp_path)
            logging.info(f"make folder")
            Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
            with open(target_path, "w") as t:
                loadme = f"""import marshal
exec(marshal.loads({a}))"""
                t.write(loadme)
    except Exception as e:
        print(e)

def main():
    parser = argparse.ArgumentParser(description='🍄')
    parser.add_argument('target_folder', type=str,
                        help='target path to obfuscate')
    parser.add_argument('--output', default='dist',  
                        type=str, dest='output_folder', nargs=1,
                        help='set ouptput folder %(prog)s (default: %(default)s)')
    
    args = parser.parse_args()
    obfuscate(args.target_folder, args.output_folder)
    logging.info("Success")

if __name__ == "__main__":
    main()