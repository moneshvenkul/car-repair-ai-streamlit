# Fine-Tuning Script for Language Agent
# Run this to fine-tune the language model for car repair Q&A

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageAgentTrainer:
    """Fine-tuning trainer for Language Agent"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.output_dir = "models/language_agent_finetuned"
        
    def create_training_dataset(self):
        """Create comprehensive car repair Q&A dataset"""
        
        # Comprehensive training data for car repair
        training_data = [
            # Battery repairs
            {
                "car": "Toyota Camry",
                "part": "Battery",
                "question": "How do I replace a car battery?",
                "answer": "For Toyota Camry battery replacement: 1. Park safely and turn off engine 2. Open hood and locate battery 3. Disconnect negative terminal first (black cable) 4. Disconnect positive terminal (red cable) 5. Remove battery hold-down bracket 6. Lift out old battery carefully 7. Clean terminals with baking soda solution 8. Install new battery in reverse order 9. Ensure tight connections. Safety: Always wear gloves and eye protection."
            },
            {
                "car": "Honda Civic",
                "part": "Battery",
                "question": "What are signs of a failing battery?",
                "answer": "Signs of failing battery in Honda Civic: slow engine cranking, dim headlights, dashboard warning lights, clicking sounds when starting, swollen battery case, corroded terminals. Test: Battery voltage should read 12.6V when engine off, 13.5-14.5V when running."
            },
            
            # Tire repairs
            {
                "car": "Ford Focus",
                "part": "Tire",
                "question": "How do I change a flat tire?",
                "answer": "Ford Focus tire change procedure: 1. Find safe location away from traffic 2. Turn on hazard lights and engage parking brake 3. Place wheel wedges behind opposite tires 4. Remove hubcap if present 5. Loosen lug nuts with lug wrench (don't remove) 6. Jack up vehicle until flat tire is 6 inches off ground 7. Remove lug nuts completely and pull tire toward you 8. Mount spare tire and replace lug nuts finger-tight 9. Lower vehicle until spare touches ground 10. Tighten lug nuts in star pattern 11. Lower vehicle completely."
            },
            {
                "car": "Nissan Altima",
                "part": "Tire",
                "question": "How do I check tire pressure?",
                "answer": "Tire pressure check for Nissan Altima: 1. Check when tires are cold (not driven for 3+ hours) 2. Remove valve cap 3. Press tire gauge firmly on valve stem 4. Read measurement 5. Compare to recommended pressure on door jamb sticker (usually 32-35 PSI) 6. Add air if needed 7. Replace valve cap. Check monthly for optimal fuel economy and tire life."
            },
            
            # Oil and maintenance
            {
                "car": "Hyundai Elantra",
                "part": "Oil Cap",
                "question": "How do I change engine oil?",
                "answer": "Hyundai Elantra oil change: 1. Warm engine slightly then turn off 2. Jack up front of car safely 3. Locate oil drain plug under engine 4. Remove oil filler cap on top of engine 5. Remove drain plug with wrench, drain oil into pan 6. Remove old oil filter, clean mounting surface 7. Apply thin layer of oil to new filter gasket 8. Install new filter hand-tight plus 3/4 turn 9. Reinstall drain plug with new gasket 10. Lower car 11. Add new oil (4.5 quarts for Elantra) 12. Check level with dipstick."
            },
            {
                "car": "Chevrolet Cruze",
                "part": "Oil Filter",
                "question": "How often should I change oil?",
                "answer": "Chevrolet Cruze oil change intervals: Conventional oil every 3,000-5,000 miles, synthetic blend every 5,000-7,500 miles, full synthetic every 7,500-10,000 miles. Check owner's manual for specific recommendations. Monitor oil color - change when it turns black or thick. Severe driving conditions (city driving, extreme temperatures) require more frequent changes."
            },
            
            # Brake system
            {
                "car": "BMW 3 Series",
                "part": "Brake Pads",
                "question": "How do I replace brake pads?",
                "answer": "BMW 3 Series brake pad replacement: 1. Jack up car and remove wheel 2. Remove brake caliper bolts 3. Lift caliper off rotor (don't let it hang by brake line) 4. Remove old brake pads 5. Push caliper piston back with C-clamp 6. Install new pads with anti-squeal compound 7. Reinstall caliper and tighten bolts to spec 8. Pump brake pedal before driving 9. Break in new pads gradually. Warning: This is advanced repair - consider professional service."
            },
            {
                "car": "Mercedes C-Class",
                "part": "Brake Rotors",
                "question": "When should I replace brake rotors?",
                "answer": "Mercedes C-Class rotor replacement indicators: thickness below minimum spec (usually stamped on rotor), deep grooves or scoring, excessive warping causing pedal pulsation, cracking, or severe rust. Measure with micrometer - replace if below 22mm thickness. Always replace rotors in pairs per axle."
            },
            
            # Electrical system
            {
                "car": "Audi A4",
                "part": "Alternator",
                "question": "What are symptoms of alternator failure?",
                "answer": "Audi A4 alternator failure symptoms: dim or flickering headlights, dashboard warning lights, dead battery, slow engine cranking, strange noises from engine bay, burning smell. Test: Engine running voltage should be 13.5-14.5V. If alternator fails, car will run on battery only until it dies."
            },
            {
                "car": "Volkswagen Jetta",
                "part": "Starter Motor",
                "question": "How do I diagnose starter problems?",
                "answer": "Volkswagen Jetta starter diagnosis: clicking sound when turning key indicates starter solenoid issue, grinding noise suggests worn starter gear, no sound at all could be electrical problem or dead starter. Check: battery voltage (12.6V), clean battery terminals, tap starter with hammer while someone turns key. If no improvement, starter replacement needed."
            },
            
            # Air system
            {
                "car": "Subaru Outback",
                "part": "Air Filter",
                "question": "How do I replace air filter?",
                "answer": "Subaru Outback air filter replacement: 1. Locate rectangular air filter housing near engine 2. Unclip or unscrew housing latches 3. Lift top of housing 4. Remove old filter, note installation direction 5. Clean housing with damp cloth 6. Install new filter ensuring proper fit 7. Close housing securely 8. Reset service interval if equipped. Replace every 12,000-15,000 miles or per schedule."
            },
            {
                "car": "Mazda 6",
                "part": "Spark Plugs",
                "question": "How do I replace spark plugs?",
                "answer": "Mazda 6 spark plug replacement: 1. Remove engine cover 2. Disconnect ignition coils from spark plugs 3. Use spark plug socket to remove old plugs 4. Gap new plugs to specification (usually 0.040-0.044 inches) 5. Thread in new plugs by hand to avoid cross-threading 6. Tighten to specification with torque wrench 7. Reconnect ignition coils 8. Replace engine cover. Replace every 30,000-100,000 miles depending on plug type."
            },
            
            # Cooling system
            {
                "car": "Kia Optima",
                "part": "Radiator",
                "question": "How do I flush radiator coolant?",
                "answer": "Kia Optima coolant flush: 1. Wait for engine to cool completely 2. Remove radiator cap 3. Drain coolant from radiator drain plug 4. Close drain plug 5. Fill with radiator flush solution and water 6. Run engine per flush instructions 7. Drain flush solution 8. Refill with proper coolant mixture (usually 50/50 coolant and water) 9. Bleed air from system 10. Check level after driving. Flush every 30,000-50,000 miles."
            },
            {
                "car": "Jeep Cherokee",
                "part": "Water Pump",
                "question": "What are signs of water pump failure?",
                "answer": "Jeep Cherokee water pump failure signs: coolant leak from front of engine, squealing noise from water pump pulley, engine overheating, steam from engine bay, coolant in oil (milky appearance). Water pump is driven by timing belt/chain, so failure can cause serious engine damage. Replace immediately if symptoms present."
            }
        ]
        
        return training_data
    
    def format_training_data(self, data):
        """Format data for language model training"""
        
        formatted_texts = []
        
        for item in data:
            # Create conversational format
            text = f"""Car: {item['car']}
Part: {item['part']}
Question: {item['question']}
Answer: {item['answer']}<|endoftext|>"""
            
            formatted_texts.append(text)
        
        return formatted_texts
    
    def create_dataset(self, texts, tokenizer, max_length=512):
        """Create tokenized dataset"""
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_model(self, epochs=3, batch_size=2, learning_rate=5e-5):
        """Fine-tune the language model"""
        
        logger.info("🚀 Starting Language Agent fine-tuning...")
        
        try:
            # Load model and tokenizer
            logger.info(f"Loading model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare training data
            logger.info("Preparing training data...")
            raw_data = self.create_training_dataset()
            formatted_texts = self.format_training_data(raw_data)
            
            logger.info(f"Created {len(formatted_texts)} training examples")
            
            # Create tokenized dataset
            train_dataset = self.create_dataset(formatted_texts, tokenizer)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=50,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                prediction_loss_only=True,
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer
            )
            
            # Train model
            logger.info("🔥 Starting training...")
            trainer.train()
            
            # Save model
            logger.info(f"💾 Saving model to {self.output_dir}")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)
            
            # Save training metadata
            metadata = {
                "model_name": self.model_name,
                "training_date": datetime.now().isoformat(),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_samples": len(formatted_texts)
            }
            
            with open(f"{self.output_dir}/training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Language Agent fine-tuning completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def test_model(self):
        """Test the fine-tuned model"""
        logger.info("🧪 Testing fine-tuned Language Agent...")
        
        try:
            # Load fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(self.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
            
            # Test queries
            test_queries = [
                ("Toyota Camry", "Battery", "How do I jump start my car?"),
                ("Honda Civic", "Brakes", "Why are my brakes squealing?"),
                ("Ford Focus", "Oil", "When should I change my oil?")
            ]
            
            for car, part, question in test_queries:
                # Create prompt
                prompt = f"Car: {car}\nPart: {part}\nQuestion: {question}\nAnswer:"
                
                # Tokenize
                inputs = tokenizer.encode(prompt, return_tensors='pt')
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False
                    )
                
                # Decode
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response.split("Answer:")[-1].strip()
                
                print(f"\n🚗 {car} | 🔧 {part}")
                print(f"❓ {question}")
                print(f"🤖 {answer}")
                print("-" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            return False

def main():
    """Main training script"""
    print("🤖 Language Agent Fine-Tuning for Car Repair")
    print("=" * 60)
    
    # Initialize trainer
    trainer = LanguageAgentTrainer()
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    # Train model
    if trainer.train_model(epochs=5, batch_size=2):
        print("\n🎉 Training completed successfully!")
        
        # Test model
        trainer.test_model()
        
        print(f"\n📁 Model saved to: {trainer.output_dir}")
        print("🚀 You can now use the fine-tuned Language Agent in your app!")
    
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()