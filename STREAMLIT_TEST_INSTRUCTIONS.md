# Streamlit App Test Instructions

## 🚀 Streamlit App is Starting!

The Streamlit app should be running at: **http://localhost:8501**

## Testing the Full Workflow with HF Transformer

### Step 1: Open the App
1. Open your web browser
2. Navigate to: **http://localhost:8501**
3. You should see the Stock Market Prediction Tool interface

### Step 2: Load Stock Data
1. In the **sidebar** (left side):
   - Enter stock symbols: `AAPL` (or `AAPL, MSFT, GOOGL` for multiple)
   - Select time period: `1y` or `2y` (longer = more data = better for HF transformer)
   - Click **"Load Data"** button
   - Wait for data to load (you'll see progress)

### Step 3: Engineer Features (Automatic)
- Features should be engineered automatically after data loads
- If not, you'll see a button to engineer features

### Step 4: Train HF Transformer Model
1. In the **sidebar**, find **"Model Settings"** section:
   - **Model Type**: Select `hf_transformer` ⭐ (This is the new option!)
   - **Prediction Task**: Select `regression`
   - **Prediction Horizon**: Select `1` (1 day ahead)
   
2. Click **"Train Model"** button

3. **What to Expect**:
   - You'll see an info message: "🤖 Using Hugging Face Transformer (PatchTST)..."
   - Training will take 2-10 minutes (depending on data size)
   - Progress will be shown
   - You'll see training loss updates

### Step 5: View Results
After training completes:
- **Metrics** will be displayed (MAE, RMSE, etc.)
- **Predictions** will be shown
- **Charts** will display actual vs predicted values

## Expected Behavior

### ✅ Success Indicators:
- Model type dropdown shows `hf_transformer` option
- Training starts without errors
- Progress updates appear
- Metrics are calculated and displayed
- Predictions are generated

### ⚠️ If You See Errors:

**"Transformers not available"**:
```bash
pip install transformers torch accelerate
```

**"Not enough data"**:
- Use longer period (2y or 5y)
- Or use multiple stocks to get more data

**"Out of memory"**:
- Close other applications
- Use smaller dataset
- Reduce context_length in code

## Testing Different Scenarios

### Test 1: Single Stock, Short Period
- Symbol: `AAPL`
- Period: `1y`
- Model: `hf_transformer`
- Horizon: `1`

### Test 2: Multiple Stocks, Longer Period
- Symbols: `AAPL, MSFT, GOOGL`
- Period: `2y`
- Model: `hf_transformer`
- Horizon: `1`

### Test 3: Compare with Other Models
- Train with `ensemble` model first
- Then train with `hf_transformer`
- Compare the metrics

## Troubleshooting

### App Not Loading?
1. Check if Streamlit is running:
   ```bash
   ps aux | grep streamlit
   ```

2. Restart the app:
   ```bash
   cd "Stock Market AI"
   python3 -m streamlit run main.py
   ```

### Model Training Fails?
1. Check console for error messages
2. Ensure you have at least 200 samples of data
3. Verify transformers is installed:
   ```bash
   python3 -c "import transformers; print('OK')"
   ```

### Slow Training?
- This is normal! HF transformer takes longer than other models
- First training will be slower (model initialization)
- Subsequent training will be faster

## What to Look For

### In the Sidebar:
- ✅ `hf_transformer` appears in Model Type dropdown
- ✅ All other options work normally

### During Training:
- ✅ Progress messages appear
- ✅ Training loss decreases
- ✅ No error messages

### After Training:
- ✅ Metrics displayed (MAE, RMSE, etc.)
- ✅ Predictions table populated
- ✅ Charts show actual vs predicted

## Next Steps After Testing

1. **Compare Performance**: Try other model types and compare metrics
2. **Experiment**: Try different prediction horizons (3, 5, 10 days)
3. **Tune Parameters**: Adjust model settings for better performance
4. **Save Models**: Models can be saved for later use

## Stopping the App

To stop the Streamlit app:
1. Press `Ctrl+C` in the terminal
2. Or close the browser tab and stop the process

---

**Happy Testing! 🎉**

If you encounter any issues, check the console output for error messages.

