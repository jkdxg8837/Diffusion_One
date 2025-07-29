stra = "named_grads"
try:
    layer_num_str = stra.split(".")[1]
except ValueError or IndexError:
    print("Index out of range")
except Exception as e:
    print(f"An error occurred: {e}")