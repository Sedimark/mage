import mlflow
import mlflow.pyfunc
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional

def load_and_predict(
    model_name: str, 
    model_version: str, 
    data: pd.DataFrame,
    validate_output: bool = True
) -> Dict[str, Any]:
    """
    Generic function to load MLFlow model and make predictions with automatic data validation
    
    Args:
        model_name: Name of the registered model
        model_version: Version of the model (e.g., "1", "latest", "staging")
        data: Input DataFrame
        validate_output: Whether to validate output against signature
    
    Returns:
        Dictionary containing predictions, validation info, and metadata
    """
    
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_info = mlflow.models.get_model_info(model_uri)
        signature = model_info.signature
        
        if signature is None:
            print("⚠️  Warning: Model has no signature. Proceeding without validation.")
            predictions = model.predict(data)
            predictions_df = _convert_predictions_to_dataframe(predictions, data)
            return {
                'predictions': predictions_df,
                'validation_passed': True,
                'input_validation': None,
                'output_validation': None,
                'model_info': {
                    'name': model_name,
                    'version': model_version,
                    'uri': model_uri
                }
            }
        
        signature_dict = signature.to_dict()
        inputs_signature = json.loads(signature_dict["inputs"])
        outputs_signature = json.loads(signature_dict.get("outputs", "[]"))
        
        print("=== Model Signature Info ===")
        print(f"Input signature: {inputs_signature}")
        if outputs_signature:
            print(f"Output signature: {outputs_signature}")
        
        # Validate and prepare input data
        input_validation_result = validate_and_prepare_input(data, inputs_signature)
        
        if not input_validation_result['success']:
            return {
                'predictions': None,
                'validation_passed': False,
                'input_validation': input_validation_result,
                'output_validation': None,
                'error': 'Input validation failed',
                'model_info': {
                    'name': model_name,
                    'version': model_version,
                    'uri': model_uri
                }
            }
        
        prepared_data = input_validation_result['prepared_data']
        print(input_validation_result["messages"])
        print(f"\nMaking predictions with data shape: {prepared_data.shape}")
        predictions = model.predict(prepared_data)
        predictions_df = _convert_predictions_to_dataframe(
            predictions, 
            data, 
            outputs_signature,
            input_validation_result
        )
        
        output_validation_result = None
        if validate_output and outputs_signature:
            output_validation_result = validate_output_against_signature(predictions, outputs_signature)
        
        return {
            'predictions': predictions_df,
            'validation_passed': True,
            'input_validation': input_validation_result,
            'output_validation': output_validation_result,
            'model_info': {
                'name': model_name,
                'version': model_version,
                'uri': model_uri,
                'signature': signature_dict
            }
        }
        
    except Exception as e:
        return {
            'predictions': None,
            'validation_passed': False,
            'error': str(e),
            'model_info': {
                'name': model_name,
                'version': model_version,
                'uri': model_uri
            }
        }

def validate_and_prepare_input(data: pd.DataFrame, inputs_signature: List[Dict]) -> Dict[str, Any]:
    """
    Validate and prepare input data against MLFlow signature
    """
    print("\n=== Input Validation ===")
    
    if len(inputs_signature) > 1:
        print(f"Model expects {len(inputs_signature)} inputs")
        return validate_multiple_inputs(data, inputs_signature)
    
    input_spec = inputs_signature[0]
    
    if input_spec.get("type") == "tensor":
        return validate_tensor_input(data, input_spec)
    elif input_spec.get("type") == "dataframe":
        return validate_dataframe_input(data, input_spec)
    else:
        return {
            'success': False,
            'messages': [f"Unsupported input type: {input_spec.get('type')}"],
            'prepared_data': None
        }

def validate_tensor_input(data: pd.DataFrame, tensor_spec: Dict) -> Dict[str, Any]:
    """Validate and prepare tensor input"""
    spec = tensor_spec.get("tensor-spec", {})
    expected_shape = spec.get("shape", [])
    expected_dtype = spec.get("dtype", "float32")
    
    messages = []
    
    target_shape = [dim for dim in expected_shape if dim != -1]
    
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) != len(data.columns):
            non_numeric = set(data.columns) - set(numeric_cols)
            messages.append(f"✗ Non-numeric columns found: {list(non_numeric)}")
            return {'success': False, 'messages': messages, 'prepared_data': None}
        
        messages.append("✓ All columns are numeric")
        
        target_elements = np.prod(target_shape) if target_shape else data.shape[1]
        
        if len(target_shape) == 0:
            prepared_data = data.values.astype(_get_numpy_dtype(expected_dtype))
        else:
            total_elements = data.size
            if total_elements % target_elements != 0:
                messages.append(f"✗ Data size ({total_elements}) not compatible with target shape ({target_shape})")
                return {'success': False, 'messages': messages, 'prepared_data': None}
            
            n_samples = total_elements // target_elements
            flat_data = data.values.flatten()
            new_shape = [n_samples] + target_shape
            prepared_data = flat_data.reshape(new_shape).astype(_get_numpy_dtype(expected_dtype))
        
        messages.append(f"✓ Reshaped to {prepared_data.shape} with dtype {prepared_data.dtype}")
        
        return {
            'success': True,
            'messages': messages,
            'prepared_data': prepared_data,
            'original_shape': data.shape,
            'final_shape': prepared_data.shape
        }
        
    except Exception as e:
        messages.append(f"✗ Preparation failed: {str(e)}")
        return {'success': False, 'messages': messages, 'prepared_data': None}

def validate_dataframe_input(data: pd.DataFrame, dataframe_spec: Dict) -> Dict[str, Any]:
    """Validate DataFrame input (for models expecting DataFrame directly)"""
    messages = []
    
    messages.append("✓ DataFrame input - using data as-is")
    
    return {
        'success': True,
        'messages': messages,
        'prepared_data': data,
        'original_shape': data.shape,
        'final_shape': data.shape
    }

def validate_multiple_inputs(data: pd.DataFrame, inputs_signature: List[Dict]) -> Dict[str, Any]:
    """Handle models with multiple inputs"""
    messages = []
    messages.append(f"⚠️  Model expects {len(inputs_signature)} inputs")
    messages.append("⚠️  Using entire DataFrame for first input - you may need to customize this")
    
    first_input = inputs_signature[0]
    return validate_and_prepare_input(data, [first_input])

def validate_output_against_signature(predictions: Any, outputs_signature: List[Dict]) -> Dict[str, Any]:
    """Validate model output against expected signature"""
    messages = []
    
    if len(outputs_signature) == 0:
        return {'success': True, 'messages': ['No output signature to validate against']}
    
    output_spec = outputs_signature[0]
    
    if output_spec.get("type") == "tensor":
        spec = output_spec.get("tensor-spec", {})
        expected_shape = spec.get("shape", [])
        expected_dtype = spec.get("dtype", "float32")
        
        if hasattr(predictions, 'shape'):
            pred_shape = predictions.shape
            pred_dtype = str(predictions.dtype) if hasattr(predictions, 'dtype') else type(predictions).__name__
            
            expected_shape_no_batch = [dim for dim in expected_shape if dim != -1]
            pred_shape_no_batch = list(pred_shape[1:]) if len(pred_shape) > 1 else list(pred_shape)
            
            if expected_shape_no_batch and pred_shape_no_batch != expected_shape_no_batch:
                messages.append(f"⚠️  Output shape mismatch: expected {expected_shape_no_batch}, got {pred_shape_no_batch}")
            else:
                messages.append(f"✓ Output shape matches: {pred_shape}")
            
            messages.append(f"✓ Output dtype: {pred_dtype}")
        else:
            messages.append("⚠️  Cannot validate output shape - predictions not array-like")
    
    return {'success': True, 'messages': messages}

def _get_numpy_dtype(dtype_string: str):
    """Convert dtype string to numpy dtype"""
    dtype_mapping = {
        'float32': np.float32,
        'float64': np.float64,
        'int32': np.int32,
        'int64': np.int64,
        'bool': bool,
        'string': str
    }
    return dtype_mapping.get(dtype_string, np.float32)


def _convert_predictions_to_dataframe(
    predictions: Any, 
    original_data: pd.DataFrame,
    outputs_signature: List[Dict] = None,
    input_validation_result: Dict = None
) -> pd.DataFrame:
    """
    Convert model predictions to a pandas DataFrame preserving original structure
    
    Args:
        predictions: Raw predictions from the model
        original_data: Original input DataFrame
        outputs_signature: MLFlow output signature (optional)
        input_validation_result: Input validation results (optional)
    
    Returns:
        pandas DataFrame with predictions
    """
    
    if isinstance(predictions, pd.DataFrame):
        print("✓ Predictions already in DataFrame format")
        return predictions
    
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    print(f"Converting predictions with shape {predictions.shape} to DataFrame")

    original_shape = original_data.shape
    original_columns = original_data.columns
    n_original_samples = original_shape[0]
    n_original_features = original_shape[1]
    
    pred_shape = predictions.shape
    
    if len(pred_shape) == 1:
        if len(predictions) == n_original_samples:
            return pd.DataFrame(predictions, columns=['prediction'], index=original_data.index)
        else:
            return pd.DataFrame(predictions, columns=['prediction'])
    
    elif len(pred_shape) == 2:
        n_pred_samples, n_pred_features = pred_shape
        
        if n_pred_samples == n_original_samples:
            
            if n_pred_features == n_original_features:
                print(f"✓ Using original column names: {list(original_columns)}")
                return pd.DataFrame(predictions, columns=original_columns, index=original_data.index)
            
            elif n_pred_features == 1:
                return pd.DataFrame(predictions, columns=['prediction'], index=original_data.index)
            
            else:
                column_names = [f'pred_{i}' for i in range(n_pred_features)]
                return pd.DataFrame(predictions, columns=column_names, index=original_data.index)
        
        else:
            if n_pred_features == n_original_features:
                column_names = list(original_columns)
            elif n_pred_features == 1:
                column_names = ['prediction']
            else:
                column_names = [f'pred_{i}' for i in range(n_pred_features)]
            
            return pd.DataFrame(predictions, columns=column_names)
    
    elif len(pred_shape) == 3:
        n_batch, n_timesteps, n_features = pred_shape
        
        if n_batch == 1:
            reshaped_preds = predictions.squeeze(0)
            print(f"✓ Removed batch dimension: {pred_shape} -> {reshaped_preds.shape}")
        
            if n_features == n_original_features:
                column_names = list(original_columns)
                print(f"✓ Using original column names: {column_names}")
            else:
                column_names = [f'pred_{i}' for i in range(n_features)]
                print(f"✓ Generated column names: {column_names}")
            
            return pd.DataFrame(reshaped_preds, columns=column_names)
        
        else:
            reshaped_preds = predictions.reshape(-1, n_features)
            print(f"✓ Reshaped multiple batches: {pred_shape} -> {reshaped_preds.shape}")
            
            if n_features == n_original_features:
                column_names = list(original_columns)
            else:
                column_names = [f'pred_{i}' for i in range(n_features)]
            
            return pd.DataFrame(reshaped_preds, columns=column_names)
    
    elif len(pred_shape) == 4:
        n_batch = pred_shape[0]
        
        if n_batch == 1:
            reshaped_preds = predictions.squeeze(0)
            if len(reshaped_preds.shape) == 3:
                h, w, c = reshaped_preds.shape
                reshaped_preds = reshaped_preds.reshape(h * w, c)
            elif len(reshaped_preds.shape) == 2:
                pass
            else:
                reshaped_preds = reshaped_preds.reshape(-1, 1)
            
            n_features = reshaped_preds.shape[1]
            if n_features == n_original_features:
                column_names = list(original_columns)
            else:
                column_names = [f'pred_{i}' for i in range(n_features)]
            
            return pd.DataFrame(reshaped_preds, columns=column_names)
        
        else:
            reshaped_preds = predictions.reshape(-1, pred_shape[-1])
            n_features = pred_shape[-1]
            
            if n_features == n_original_features:
                column_names = list(original_columns)
            else:
                column_names = [f'pred_{i}' for i in range(n_features)]
            
            return pd.DataFrame(reshaped_preds, columns=column_names)
    
    else:
        if pred_shape[0] == 1:
            reshaped_preds = predictions.squeeze(0)
            if len(reshaped_preds.shape) > 2:
                reshaped_preds = reshaped_preds.reshape(-1, reshaped_preds.shape[-1])
            elif len(reshaped_preds.shape) == 1:
                reshaped_preds = reshaped_preds.reshape(-1, 1)
        else:
            reshaped_preds = predictions.reshape(-1, pred_shape[-1])
        
        n_features = reshaped_preds.shape[1]
        if n_features == n_original_features:
            column_names = list(original_columns)
        else:
            column_names = [f'pred_{i}' for i in range(n_features)]
        
        return pd.DataFrame(reshaped_preds, columns=column_names)