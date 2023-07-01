from resfpn34.models.model import Resfpn34

Resfpn34.initialize(num_threads=1)  # Initialize
npmodel = Resfpn34()
image_path = "./test.jpg"  #Image path
print(npmodel.run(image_path))  # Inference
Resfpn34.finalize()  # Memory management