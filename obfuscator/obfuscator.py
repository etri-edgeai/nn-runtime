import marshal
import os
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
    """난독화 작업을 하는 함수입니다. target folder의 *.py 파일들의 난독화를 진행합니다.
    Parameters
    ----------
    str: target_folder
    str: output_folder, default="dist"
    """
    result = []
    for j in ["*.py"]:
        result.extend([y for x in os.walk(target_folder) for y in glob(os.path.join(x[0], j))])

    logging.info(f"make folder")
    os.makedirs(os.path.join(output_folder, target_folder),exist_ok=True)

    for i in result:
        logging.info(f"{i} file obfuscate")
        with open(i, "rb") as f:
            code = compile(f.read(), "", mode='exec', dont_inherit=True)
            py_bytes = marshal.dumps(code)
            a = str(py_bytes)

        with open(os.path.join(output_folder, i), "w") as f:
            loadme = f"""import marshal
exec(marshal.loads({a}))"""
            f.write(loadme)

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