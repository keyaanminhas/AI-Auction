

# **Regatta AI Competition**

This project was created for the **Regatta Competition**, where the challenge involved developing an AI-powered system for object recognition and an optimized bidding strategy. The system identifies objects from images using a trained CNN model and places bids intelligently while adhering to a predefined budget and item requirements.

---

## **Project Overview**

### **Objective**
1. Use AI to identify specific objects and determine their classification with high accuracy.
2. Develop a bidding strategy that balances cost-effectiveness with competition dynamics.
3. Manage resources and avoid overbidding while maximizing the overall outcome.

---

## **AI Model: Object Detection**

### **1. Architecture**
The core AI system uses a **Convolutional Neural Network (CNN)** based on **MobileNetV2**, a lightweight and efficient architecture widely used for mobile and embedded applications.

#### **Key Components**
- **Base Model**: MobileNetV2 pretrained on ImageNet was used as the feature extractor. The pre-trained layers were frozen to retain generic features like edges, patterns, and textures.
- **Custom Layers**:
  - **Data Augmentation**: Introduced transformations such as random flips, rotations, zooms, and translations to simulate real-world scenarios and improve generalization.
  - **Global Average Pooling (GAP)**: Aggregates spatial information to reduce the number of trainable parameters.
  - **Fully Connected Dense Layers**:
    - **128-unit Dense Layer** with L2 regularization to prevent overfitting.
    - **7-unit Output Layer** with a Softmax activation function for multi-class classification (7 object types).

### **2. Training Process**
- **Dataset**: The dataset was curated under the assumption that a camera feed would be used during the competition. Images of seven object classes were collected and labeled:
  - **Bamboo**
  - **Stringball**
  - **Juterope**
  - **Tyretube**
  - **Barrel**
  - **Oar**
  - **Ladder**
- **Data Augmentation**: Augmentation techniques like flips, rotations, zooms, and translations were applied to improve model robustness and prevent overfitting.
- **Loss Function**: Sparse Categorical Crossentropy was used to measure the difference between predicted and actual labels.
- **Optimizer**: Adam Optimizer with a learning rate of 0.0003 for faster convergence.
- **Metrics**: Accuracy and validation loss were tracked to monitor model performance.
- **Training**:
  - 40 epochs of training with a validation split.
  - Early stopping and best-model checkpointing ensured only the most accurate model was retained.
 
  - ![Alt text](./results.jpg)

### **3. Results**
- **Performance**:
  - High training and validation accuracy, with minimal overfitting due to dropout and L2 regularization.
  - Robust performance on unseen test images, with consistent classification across all object classes.
- **Model Outputs**: Predictions are probabilities for each of the seven classes, with the highest probability determining the final classification.

### **4. Model Deployment**
The trained model was saved in two formats:
- `.keras`: For compatibility with TensorFlow 2.x frameworks.
- `.h5`: For lightweight deployment and ease of integration into production pipelines.

---

## **Bidding System**

### **Core Principles**
1. **Risk Management**:
   - Avoid overbidding by capping bids at maximum prices.
   - Manage inventory by ensuring no over-purchasing of any item.
2. **Budget Management**:
   - Prevent bids that exceed the remaining budget.
   - Prioritize items based on strategic importance and availability.
3. **Dynamic Bidding**:
   - Place low bids (20–50% of the max price) for items no longer required to discourage competitors without losing credibility.
   - Opportunistically bid on underpriced items for potential resale.
4. **Competitor Influence**:
   - Analyze competitors' highest bids to adjust future bid prices dynamically.

### **Bidding Workflow**
1. **Bid Placement**:
   - For required items, bids are placed within their price constraints and adjusted to avoid exceeding the budget.
   - For non-required items, lower bids are placed to create a psychological effect on competitors.
2. **Bid Evaluation**:
   - Check the highest bidder and adjust the remaining budget and item inventory accordingly.
   - Update prices dynamically based on competitors' bid history.
3. **Result Logging**:
   - Automatically log bid details in Google Sheets, including timestamps and item details.

### **Algorithm Features**
- **Auto-Counter**: Tracks inventory levels to avoid overbuying.
- **Budget Cap**: Ensures bids are within budgetary constraints.
- **Strategic Low Bidding**: Places lower bids for non-priority items to influence competitors' behavior.
- **Max Price Enforcement**: Caps bids at a predefined maximum for each item.
- **Opportunistic Resale**: Purchases underpriced items for potential resale at higher values.
- **Google Sheets Integration**: Logs bid data and retrieves the highest bid for efficient management.

---

## **Features**

- **AI-Powered Object Recognition**: Efficient classification using a trained MobileNetV2 model.
- **Budget-Conscious Bidding**: Ensures no overspending or over-purchasing.
- **Dynamic Pricing Adjustment**: Reacts to competitors' bids to stay competitive.
- **Resale Strategy**: Buys cheap items to sell later at higher prices.
- **Google Sheets Integration**: Automates bid logging and tracking for real-time insights.

---

## **Usage Instructions**

1. **Object Detection**:
   - Provide an image to the `detect_image()` function.
   - The function returns the classified object name.

2. **Bidding**:
   - Use the `make_bid(item, bid)` function to place a bid.
   - Ensure the Excel file (`Team5.xlsx`) is accessible for bid logging.

3. **Highest Bidder Retrieval**:
   - Use the `highest_bidder(bid_num)` function to fetch the highest bid details for a specific bid round.

---

## **Requirements**

- **Python 3.7+**
- Libraries:
  - TensorFlow (tested with version 2.x)
  - NumPy
  - Pillow
  - OpenCV
  - OpenPyXL
- **Google Drive Integration**:
  - Ensure Google Drive is mounted for accessing `Team5.xlsx`.

---

## **Future Improvements**

- Transition to an object-oriented design for better code maintainability.
- Introduce real-time bidding dynamics using live feeds.
- Expand the dataset to improve object detection accuracy.
- Enhance bidding algorithms with advanced machine learning techniques for predictive competitor analysis.

---
