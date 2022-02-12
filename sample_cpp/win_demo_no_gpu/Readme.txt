Under win os, open cmd prompt in this directory, then type :

sample_cpp -focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx -global_view_model=../../data/models/lpreditor_anpr_global_view.onnx -dir=../../data/images/benchmarks-master/endtoend/plate_un -plates_types_model=../../data/models/plates_types_7.onnx -plates_types_labels=../../data/models/plates_types_7.txt
or
sample_cpp -focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx -global_view_model=../../data/models/lpreditor_anpr_global_view.onnx -image=../../data/images/0000000001_3065WWA34.jpg -plates_types_model=../../data/models/plates_types_7.onnx -plates_types_labels=../../data/models/plates_types_7.txt
or
sample_cpp -focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx -global_view_model=../../data/models/lpreditor_anpr_global_view.onnx -image=path/to/your/image/file -plates_types_model=../../data/models/plates_types_7.onnx -plates_types_labels=../../data/models/plates_types_7.txt