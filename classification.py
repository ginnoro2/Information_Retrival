import tkinter as tk
import tkinter.ttk as ttk  # Add this line to import ttk module
from tkinter import messagebox, scrolledtext, filedialog, simpledialog  # Added simpledialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from PIL import Image, ImageTk  # Added for image handling

class AdvancedDocumentClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Document Classification System")
        master.geometry("1000x800")
        master.configure(bg='#f0f0f0')

        # Predefined dataset (you can expand or modify these)
        self.predefined_datasets = {
            'Politics': [
                "The government announced new economic policies today.",
                "Parliamentary debate heated up over budget allocations.",
                "Election campaign strategies are being finalized.",
                "Political leaders meet to discuss national security.",
                "New legislation proposed to address climate change.",
                "The U.S. government unveils a $2 trillion infrastructure plan to boost economic growth.",
                "Senators debate the impact of proposed tax reforms on small businesses.",
                "The presidential candidates outline their foreign policy strategies ahead of the debate.",
                "International leaders convene to discuss global climate action agreements.",
                "A new bill aimed at strengthening national cybersecurity passes the first round of approval."

            ],
            'Business': [
                "Tech startup raises $50 million in series A funding.",
                "Stock market shows positive growth this quarter.",
                "Major corporation announces merger plans.",
                "New innovative product launch expected next month.",
                "Global trade negotiations continue to evolve.",
                "Tesla's stock price surges after record-breaking electric vehicle sales.",
                "Apple announces plans to acquire a leading AI startup for $1.5 billion.",
                "Small businesses face challenges adapting to new digital payment regulations.",
                "The global oil market experiences volatility amid supply chain disruptions.",
                "E-commerce giants report unprecedented holiday season revenue growth."

            ],
            'Health': [
                "New medical research breakthrough in cancer treatment.",
                "Importance of mental health awareness highlighted.",
                "Vaccination program expands to new regions.",
                "Nutrition experts recommend balanced diet strategies.",
                "Latest developments in telemedicine technologies.",
                "A groundbreaking study finds a new potential treatment for Alzheimer's disease.",
                "The CDC issues updated guidelines on COVID-19 booster shots for high-risk individuals.",
                "A surge in mental health disorders among teenagers prompts urgent policy discussions.",
                "New research highlights the long-term effects of processed food consumption on gut health.",
                "Telemedicine adoption rises as hospitals embrace remote patient monitoring technologies."

            ]
        }

        # Create main notebook
        self.notebook = tk.ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Create tabs
        self.train_tab = tk.Frame(self.notebook)
        self.classify_tab = tk.Frame(self.notebook)
        self.visualization_tab = tk.Frame(self.notebook)

        self.notebook.add(self.train_tab, text="Train Model")
        self.notebook.add(self.classify_tab, text="Classify Document")
        self.notebook.add(self.visualization_tab, text="Model Visualization")

        # Initialize classifier components
        self.classifier = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.categories = ['Politics', 'Business', 'Health']
        
        # Setup tabs
        self.setup_training_tab()
        self.setup_classification_tab()
        self.setup_visualization_tab()

    def setup_training_tab(self):
        # Title
        tk.Label(self.train_tab, text="Document Classification Training", 
                 font=('Arial', 14, 'bold')).pack(pady=10)

        # Predefined Dataset Frame
        dataset_frame = tk.Frame(self.train_tab)
        dataset_frame.pack(pady=10)

        tk.Label(dataset_frame, text="Predefined Datasets:", font=('Arial', 12)).pack()
        
        for category in self.categories:
            tk.Button(dataset_frame, text=f"Load {category} Dataset", 
                      command=lambda cat=category: self.load_predefined_dataset(cat)
                      ).pack(side=tk.LEFT, padx=5)

        # Custom Dataset Button
        tk.Button(dataset_frame, text="Load Custom Documents", 
                  command=self.load_custom_documents).pack(side=tk.LEFT, padx=5)

        # Training Output
        self.training_output = scrolledtext.ScrolledText(self.train_tab, height=10, width=80)
        self.training_output.pack(pady=10)

        # Train Model Button
        tk.Button(self.train_tab, text="Train Model", command=self.train_model, 
                  bg='green', fg='white').pack(pady=10)

    def setup_classification_tab(self):
        # Document Input
        tk.Label(self.classify_tab, text="Enter Document to Classify", 
                 font=('Arial', 12, 'bold')).pack(pady=10)

        self.doc_input = scrolledtext.ScrolledText(self.classify_tab, height=10, width=80)
        self.doc_input.pack(pady=10)

        # Classify Button
        tk.Button(self.classify_tab, text="Classify", command=self.classify_document, 
                  bg='blue', fg='white').pack(pady=10)

        # Classification Results
        self.classification_output = scrolledtext.ScrolledText(self.classify_tab, height=10, width=80)
        self.classification_output.pack(pady=10)

    def setup_visualization_tab(self):
        # Placeholder for visualization components
        tk.Label(self.visualization_tab, text="Model Performance Visualization", 
                 font=('Arial', 14, 'bold')).pack(pady=10)

        # Button to generate visualizations
        tk.Button(self.visualization_tab, text="Generate Visualizations", 
                  command=self.generate_visualizations, 
                  bg='purple', fg='white').pack(pady=10)

        # Visualization output area
        self.visualization_frame = tk.Frame(self.visualization_tab)
        self.visualization_frame.pack(expand=True, fill='both', padx=10, pady=10)

    def load_predefined_dataset(self, category):
        """Load predefined dataset for a category"""
        documents = self.predefined_datasets[category]
        setattr(self, f'{category.lower()}_docs', documents)
        self.training_output.insert(tk.END, f"Loaded {len(documents)} predefined {category} documents\n")

    def load_custom_documents(self):
        """Load custom documents from files"""
        category = tk.simpledialog.askstring("Input", "Enter document category (Politics/Business/Health):")
        if category not in self.categories:
            messagebox.showerror("Error", "Invalid category!")
            return

        filetypes = [('Text Files', '*.txt'), ('All Files', '*.*')]
        files = filedialog.askopenfilenames(title=f"Select {category} Documents", 
                                            filetypes=filetypes)
        
        documents = []
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
            except Exception as e:
                messagebox.showerror("Error", f"Could not read {file}: {str(e)}")
        
        setattr(self, f'{category.lower()}_docs', documents)
        self.training_output.insert(tk.END, f"Loaded {len(documents)} custom {category} documents\n")

    def preprocess_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def train_model(self):
        """Train the classification model"""
        try:
            # Collect documents from all categories
            documents = []
            labels = []

            for category in self.categories:
                docs = getattr(self, f'{category.lower()}_docs', [])
                documents.extend(docs)
                labels.extend([category] * len(docs))

            if not documents:
                messagebox.showwarning("Warning", "No documents loaded!")
                return

            # Preprocess documents
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_docs, labels, test_size=0.2, random_state=42
            )
            
            # Vectorize text
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            X_test_vectorized = self.vectorizer.transform(X_test)
            
            # Train classifier
            self.classifier = MultinomialNB()
            self.classifier.fit(X_train_vectorized, y_train)
            
            # Evaluate model
            y_pred = self.classifier.predict(X_test_vectorized)
            report = classification_report(y_test, y_pred)
            
            # Display results
            self.training_output.delete(1.0, tk.END)
            self.training_output.insert(tk.END, "Model Training Complete!\n\n")
            self.training_output.insert(tk.END, "Classification Report:\n")
            self.training_output.insert(tk.END, report)
            
            # Store test data for visualization
            self.test_data = {
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            messagebox.showinfo("Training", "Model trained successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def classify_document(self):
        """Classify a new document"""
        if not self.classifier:
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        # Get document from input
        document = self.doc_input.get(1.0, tk.END).strip()
        
        if not document:
            messagebox.showwarning("Warning", "Please enter a document to classify!")
            return

        # Preprocess and classify
        processed_doc = self.preprocess_text(document)
        vectorized_doc = self.vectorizer.transform([processed_doc])
        
        prediction = self.classifier.predict(vectorized_doc)
        probabilities = self.classifier.predict_proba(vectorized_doc)
        
        # Clear previous output
        self.classification_output.delete(1.0, tk.END)
        
        # Display results
        self.classification_output.insert(tk.END, "Classification Result:\n")
        self.classification_output.insert(tk.END, f"Predicted Category: {prediction[0]}\n\n")
        
        self.classification_output.insert(tk.END, "Probability Distribution:\n")
        for category, prob in zip(self.categories, probabilities[0]):
            self.classification_output.insert(tk.END, f"{category}: {prob:.2%}\n")

    def generate_visualizations(self):
        """Generate and display model performance visualizations"""
        if not hasattr(self, 'test_data'):
            messagebox.showwarning("Warning", "Train the model first!")
            return

        # Clear previous visualizations
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.test_data['y_test'], self.test_data['y_pred'], 
                               labels=self.categories)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.categories, 
                    yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Pie Chart of Classification Distribution
        plt.figure(figsize=(8, 6))
        unique, counts = np.unique(self.test_data['y_pred'], return_counts=True)
        plt.pie(counts, labels=unique, autopct='%1.1f%%')
        plt.title('Classification Distribution')
        plt.tight_layout()
        plt.savefig('classification_distribution.png')

        # Display images in Tkinter
        from PIL import Image, ImageTk
        
        # Confusion Matrix Image
        cm_image = Image.open('confusion_matrix.png')
        cm_photo = ImageTk.PhotoImage(cm_image)
        cm_label = tk.Label(self.visualization_frame, image=cm_photo)
        cm_label.image = cm_photo  # Keep a reference
        cm_label.pack(side=tk.LEFT, padx=10)

        # Classification Distribution Image
        dist_image = Image.open('classification_distribution.png')
        dist_photo = ImageTk.PhotoImage(dist_image)
        dist_label = tk.Label(self.visualization_frame, image=dist_photo)
        dist_label.image = dist_photo  # Keep a reference
        dist_label.pack(side=tk.LEFT, padx=10)

def main():
    root = tk.Tk()
    app = AdvancedDocumentClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()