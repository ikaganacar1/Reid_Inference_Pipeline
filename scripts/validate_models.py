#!/usr/bin/env python3                                                                                               
"""                                                                                                                  
  Model Validation Script                                                                                              
  Validates YOLO and TAO ReID models are properly set up                                                               
"""                                                                                                                  
                                                                                                                       
import argparse                                                                                                      
import sys                                                                                                           
from pathlib import Path                                                                                             
                                                                                                                   
import yaml                                                                                                          
                                                                                                                   
# Add parent to path                                                                                                 
sys.path.append(str(Path(__file__).parent.parent))                                                                   
                                                                                                                   
                                                                                                                   
def load_reid_config():                                                                                              
  """Load ReID config"""                                                                                           
  config_path = Path("configs/reid_config.yaml")                                                                   
  with open(config_path) as f:                                                                                     
      return yaml.safe_load(f)                                                                                     
                                                                                                                   
                                                                                                                   
def validate_yolo_model():                                                                                           
  """Validate YOLO model"""                                                                                        
  print("="*60)                                                                                                    
  print("Validating YOLO Model")                                                                                   
  print("="*60)                                                                                                    
                                                                                                                   
  model_path = Path("models/yolo11n.pt")                                                                           
  if not model_path.exists():                                                                                      
      print(f"✗ YOLO model not found: {model_path}")                                                               
      return False                                                                                                 
                                                                                                                   
  print(f"✓ YOLO model found: {model_path}")                                                                       
  print(f"  Size: {model_path.stat().st_size / (1024**2):.2f} MB")                                                 
                                                                                                                   
  try:                                                                                                             
      from ultralytics import YOLO                                                                                 
      import numpy as np                                                                                           
                                                                                                                   
      model = YOLO(str(model_path))                                                                                
      print("✓ YOLO model loaded successfully")                                                                    
                                                                                                                   
      dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)                                         
      results = model(dummy_img, conf=0.5, classes=[0], verbose=False)                                             
                                                                                                                   
      print(f"✓ YOLO inference test passed")                                                                       
      print(f"  Detected {len(results[0].boxes)} objects")                                                         
                                                                                                                   
      return True                                                                                                  
                                                                                                                   
  except Exception as e:                                                                                           
      print(f"✗ YOLO validation failed: {e}")                                                                      
      return False                                                                                                 
                                                                                                                   
                                                                                                                   
def validate_triton_server(config):                                                                                  
  """Validate Triton server connection"""                                                                          
  print("\n" + "="*60)                                                                                             
  print("Validating Triton Inference Server")                                                                      
  print("="*60)                                                                                                    
                                                                                                                   
  try:                                                                                                             
      import tritonclient.http as httpclient                                                                       
                                                                                                                   
      triton_url = config['triton']['server_url']                                                                  
      model_name = config['triton']['model_name']                                                                  
                                                                                                                   
      client = httpclient.InferenceServerClient(url=triton_url, verbose=False)                                     
                                                                                                                   
      if not client.is_server_live():                                                                              
          print("✗ Triton server is not live")                                                                     
          print("  Start server with: bash scripts/start_triton_server.sh")                                        
          return False                                                                                             
                                                                                                                   
      print("✓ Triton server is live")                                                                             
                                                                                                                   
      if not client.is_server_ready():                                                                             
          print("✗ Triton server is not ready")                                                                    
          return False                                                                                             
                                                                                                                   
      print("✓ Triton server is ready")                                                                            
                                                                                                                   
      if not client.is_model_ready(model_name, "1"):                                                               
          print(f"✗ Model '{model_name}' is not loaded")                                                           
          print(f"  Check triton_models/{model_name}/1/ has model.onnx")                                           
          return False                                                                                             
                                                                                                                   
      print(f"✓ Model '{model_name}' is loaded and ready")                                                         
                                                                                                                   
      metadata = client.get_model_metadata(model_name, "1")                                                        
      print(f"  Model version: {metadata.get('versions', [])}")                                                    
                                                                                                                   
      return True                                                                                                  
                                                                                                                   
  except Exception as e:                                                                                           
      print(f"✗ Triton validation failed: {e}")                                                                    
      print("  Make sure Triton server is running")                                                                
      return False                                                                                                 
                                                                                                                   
                                                                                                                   
def validate_onnx_model(config):                                                                                     
  """Validate ONNX model"""                                                                                        
  print("\n" + "="*60)                                                                                             
  print("Validating ONNX Model")                                                                                   
  print("="*60)                                                                                                    
                                                                                                                   
  onnx_path = Path(config['model']['onnx_path'])                                                                   
  if not onnx_path.exists():                                                                                       
      print(f"✗ ONNX model not found: {onnx_path}")                                                                
      return False                                                                                                 
                                                                                                                   
  print(f"✓ ONNX model found: {onnx_path}")                                                                        
  print(f"  Size: {onnx_path.stat().st_size / (1024**2):.2f} MB")                                                  
                                                                                                                   
  try:                                                                                                             
      import onnx                                                                                                  
      model = onnx.load(str(onnx_path))                                                                            
      onnx.checker.check_model(model)                                                                              
                                                                                                                   
      print("✓ ONNX model is valid")                                                                               
                                                                                                                   
      for input_tensor in model.graph.input:                                                                       
          shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]                               
          print(f"  Input '{input_tensor.name}': {shape}")                                                         
                                                                                                                   
      for output_tensor in model.graph.output:                                                                     
          shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]                              
          print(f"  Output '{output_tensor.name}': {shape}")                                                       
                                                                                                                   
      return True                                                                                                  
                                                                                                                   
  except Exception as e:                                                                                           
      print(f"✗ ONNX validation failed: {e}")                                                                      
      return False                                                                                                 
                                                                                                                   
                                                                                                                   
def validate_triton_model_files(config):                                                                             
  """Validate Triton model files exist"""                                                                          
  print("\n" + "="*60)                                                                                             
  print("Validating Triton Model Files")                                                                           
  print("="*60)                                                                                                    
                                                                                                                   
  model_name = config['triton']['model_name']                                                                      
  model_dir = Path(f"triton_models/{model_name}/1")                                                                
  config_file = Path(f"triton_models/{model_name}/config.pbtxt")                                                   
                                                                                                                   
  # Check for ONNX model (not TensorRT)                                                                            
  model_file = model_dir / "model.onnx"                                                                            
                                                                                                                   
  if not config_file.exists():                                                                                     
      print(f"✗ Triton config not found: {config_file}")                                                           
      return False                                                                                                 
  print(f"✓ Triton config found: {config_file}")                                                                   
                                                                                                                   
  if not model_file.exists():                                                                                      
      print(f"✗ Model file not found: {model_file}")                                                               
      print(f"  Copy with: cp {config['model']['onnx_path']} {model_file}")                                        
      return False                                                                                                 
                                                                                                                   
  print(f"✓ Model file found: {model_file}")                                                                       
  print(f"  Size: {model_file.stat().st_size / (1024**2):.2f} MB")                                                 
                                                                                                                   
  return True                                                                                                      
                                                                                                                   
                                                                                                                   
def main():                                                                                                          
  parser = argparse.ArgumentParser(description="Validate pipeline models")                                         
  parser.add_argument("--skip-triton", action="store_true", help="Skip Triton server check")                       
  args = parser.parse_args()                                                                                       
                                                                                                                   
  print("\n" + "="*60)                                                                                             
  print("Pipeline Model Validation")                                                                               
  print("="*60)                                                                                                    
                                                                                                                   
  config = load_reid_config()                                                                                      
  results = {}                                                                                                     
                                                                                                                   
  results['yolo'] = validate_yolo_model()                                                                          
  results['onnx'] = validate_onnx_model(config)                                                                    
  results['triton_files'] = validate_triton_model_files(config)                                                    
                                                                                                                   
  if not args.skip_triton:                                                                                         
      results['triton_server'] = validate_triton_server(config)                                                    
                                                                                                                   
  print("\n" + "="*60)                                                                                             
  print("Validation Summary")                                                                                      
  print("="*60)                                                                                                    
                                                                                                                   
  all_passed = True                                                                                                
  for name, passed in results.items():                                                                             
      status = "✓ PASS" if passed else "✗ FAIL"                                                                    
      print(f"{status}: {name.upper()}")                                                                           
      if not passed:                                                                                               
          all_passed = False                                                                                       
                                                                                                                   
  if all_passed:                                                                                                   
      print("\n✓ All validations passed! Ready to run pipeline.")                                                  
      return 0                                                                                                     
  else:                                                                                                            
      print("\n✗ Some validations failed. Please fix issues above.")                                               
      return 1                                                                                                     
                                                                                                                   
                                                                                                                   
if __name__ == "__main__":                                                                                           
  sys.exit(main()) 
