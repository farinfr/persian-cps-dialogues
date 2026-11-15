#!/usr/bin/env python3
"""
Dialogue Template Generator
===========================

Generates simple template-based dialogues from sampled products:
- Round 1: User asks question based on title, brand, category
- Rounds 2-6/7: Assistant asks about sampled features, user responds with values
"""

import json
import random
from typing import List, Dict, Any

class DialogueGenerator:
    def __init__(self, sampled_products_file: str):
        self.sampled_products_file = sampled_products_file
        self.samples = []
        self.dialogues = []
        
        # Search templates (user is searching for a product)
        self.user_search_templates = [
            "می‌خواهم محصولی در دسته {category} پیدا کنم.",
            "به دنبال محصولی در گروه {category} هستم.",
            "آیا محصولی در دسته‌بندی {category} دارید؟",
            "می‌خواهم محصولی در {category} جستجو کنم.",
            "لطفا محصولی در دسته {category} به من نشان دهید.",
            "من به دنبال محصولی در دسته‌بندی {category} می‌گردم.",
        ]
        
        self.assistant_search_response_templates = [
            "بله، در دسته {category} محصولات خوبی داریم. برای کمک بهتر، لطفا {feature} مورد نظر خود را بگویید.",
            "حتما. در گروه {category} محصولات متنوعی موجود است. می‌توانید {feature} مورد نظر خود را مشخص کنید؟",
            "خوشحالم که می‌توانم کمک کنم. برای یافتن محصول مناسب در {category}، لطفا {feature} را مشخص کنید.",
            "بله، در خدمت شما هستم. در دسته {category} محصولات مختلفی داریم. در مورد {feature} چه می‌خواهید؟",
            "بسیار خوشحالم. برای پیدا کردن محصول مناسب در {category}، لطفا {feature} مورد نظر را بگویید.",
        ]
        
        self.assistant_feature_questions = [
            "در مورد {feature} چه می‌خواهید؟",
            "آیا {feature} خاصی مد نظر دارید؟",
            "لطفا {feature} مورد نظر خود را بگویید.",
            "برای یافتن محصول مناسب، {feature} را مشخص کنید.",
            "در مورد {feature} چه ترجیحی دارید؟",
        ]
        
        self.user_answer_templates = [
            "می‌خواهم {feature} {value} باشد.",
            "من به دنبال محصولی با {feature} {value} می‌گردم.",
            "ترجیح می‌دهم {feature} {value} باشد.",
            "آیا محصولی با {feature} {value} دارید؟",
            "من می‌خواهم {feature} {value} باشد.",
            "محصولی با {feature} {value} می‌خواهم.",
        ]
        
        self.assistant_follow_up_templates = [
            "بسیار خوب، پس {feature} {value} می‌خواهید. در مورد {next_feature} چه می‌خواهید؟",
            "عالی، پس {feature} {value} است. برای پیدا کردن محصول بهتر، لطفا {next_feature} را هم بگویید.",
            "درست است. {feature} {value} را ثبت کردم. در مورد {next_feature} چه می‌خواهید؟",
            "بله، {feature} {value} را ثبت کردم. لطفا {next_feature} مورد نظر را هم بگویید.",
            "خوب، پس {feature} {value} می‌خواهید. در مورد {next_feature} چه می‌خواهید؟",
        ]
        
        self.assistant_confirmation_templates = [
            "بله، {feature} {value} را ثبت کردم.",
            "عالی، {feature} {value} را یادداشت کردم.",
            "بسیار خوب، پس {feature} {value} است.",
            "درست است، {feature} {value} را ثبت کردم.",
            "خوب، {feature} {value} را یادداشت کردم.",
        ]
        
        self.closing_templates = [
            "عالی بود. بر اساس مشخصاتی که گفتید، {title} از برند {brand} می‌تواند مناسب شما باشد. اگر سوال دیگری دارید، خوشحال می‌شوم کمک کنم.",
            "بسیار خوشحالم. بر اساس نیازهایی که گفتید، {title} {brand} را پیشنهاد می‌کنم. اگر اطلاعات بیشتری نیاز دارید، در خدمت هستم.",
            "خوب، بر اساس ویژگی‌هایی که مشخص کردید، {title} از {brand} می‌تواند گزینه مناسبی باشد. سوال دیگری دارید؟",
            "بر اساس ترجیحاتی که گفتید، {title} {brand} را پیشنهاد می‌دهم. امیدوارم مفید بوده باشد.",
            "عالی، با توجه به مشخصاتی که گفتید، {title} از برند {brand} می‌تواند برای شما مناسب باشد. اگر نیاز به اطلاعات بیشتری دارید، بفرمایید.",
        ]
    
    def load_samples(self):
        """Load sampled products"""
        print(f"Loading sampled products from {self.sampled_products_file}...")
        with open(self.sampled_products_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'samples' in data:
            self.samples = data['samples']
        else:
            self.samples = data
        
        print(f"Loaded {len(self.samples)} product samples")
        return self.samples
    
    def generate_dialogue(self, sample: Dict[str, Any], num_rounds: int = None) -> List[Dict[str, str]]:
        """
        Generate a dialogue for a product sample
        
        Args:
            sample: Product sample with features
            num_rounds: Number of dialogue rounds (6 or 7), if None randomly chooses
            
        Returns:
            List of dialogue turns
        """
        if 'error' in sample:
            return []
        
        product = sample.get('product', {})
        title = product.get('title', 'محصول')
        brand = product.get('brand', 'برند')
        category = product.get('category_name', 'دسته‌بندی')
        
        sampled_features = sample.get('sampled_features', {})
        feature_order = sample.get('feature_sampling_order', list(sampled_features.keys()))
        
        if not sampled_features:
            return []
        
        # Determine number of rounds (14)
        if num_rounds is None:
            num_rounds = 14
        
        # Pattern for 14 rounds with 5 features:
        # 1. User search -> 2. Asst Q (feature 1) -> 3. User A (feature 1) ->
        # 4. Asst confirm + Q (feature 2) -> 5. User A (feature 2) ->
        # 6. Asst confirm + Q (feature 3) -> 7. User A (feature 3) ->
        # 8. Asst confirm + Q (feature 4) -> 9. User A (feature 4) ->
        # 10. Asst confirm + Q (feature 5) -> 11. User A (feature 5) ->
        # 12. Asst confirm -> 13-14. Asst closing
        
        if num_rounds == 14:
            # 14 rounds: 5 features
            features_to_discuss = feature_order[:min(5, len(feature_order))]
            
            dialogue = []
            
            # Round 1: User searches for a product in category
            user_template = random.choice(self.user_search_templates)
            user_search = user_template.format(category=category)
            dialogue.append({'role': 'user', 'text': user_search})
            
            # Process each feature (5 features)
            for i, feature_name in enumerate(features_to_discuss):
                feature_value = sampled_features.get(feature_name, '')
                feature_display = feature_name.split('_', 1)[-1] if '_' in feature_name else feature_name
                
                if i == 0:
                    # Round 2: First assistant response - asks about feature 1
                    assistant_template = random.choice(self.assistant_search_response_templates)
                    assistant_question = assistant_template.format(
                        category=category,
                        feature=feature_display
                    )
                    dialogue.append({'role': 'assistant', 'text': assistant_question})
                else:
                    # Round 4, 6, 8, 10: Assistant confirms previous feature and asks about next
                    prev_feature_name = features_to_discuss[i-1]
                    prev_feature_value = sampled_features.get(prev_feature_name, '')
                    prev_feature_display = prev_feature_name.split('_', 1)[-1] if '_' in prev_feature_name else prev_feature_name
                    
                    follow_up_template = random.choice(self.assistant_follow_up_templates)
                    assistant_follow_up = follow_up_template.format(
                        feature=prev_feature_display,
                        value=prev_feature_value,
                        next_feature=feature_display
                    )
                    dialogue.append({'role': 'assistant', 'text': assistant_follow_up})
                
                # Round 3, 5, 7, 9, 11: User answers feature
                user_answer_template = random.choice(self.user_answer_templates)
                user_answer = user_answer_template.format(
                    feature=feature_display,
                    value=feature_value
                )
                dialogue.append({'role': 'user', 'text': user_answer})
            
            # Round 12: Final confirmation of last feature
            last_feature_name = features_to_discuss[-1]
            last_feature_value = sampled_features.get(last_feature_name, '')
            last_feature_display = last_feature_name.split('_', 1)[-1] if '_' in last_feature_name else last_feature_name
            
            confirmation_template = random.choice(self.assistant_confirmation_templates)
            final_confirmation = confirmation_template.format(
                feature=last_feature_display,
                value=last_feature_value
            )
            dialogue.append({'role': 'assistant', 'text': final_confirmation})
            
            # Round 13: First closing message
            closing_template1 = random.choice(self.closing_templates)
            closing1 = closing_template1.format(title=title, brand=brand)
            dialogue.append({'role': 'assistant', 'text': closing1})
            
            # Round 14: Second closing message
            closing_template2 = random.choice([
                "اگر سوال دیگری دارید یا نیاز به اطلاعات بیشتری دارید، خوشحال می‌شوم کمک کنم.",
                "امیدوارم این محصول بتواند نیاز شما را برطرف کند. در صورت نیاز به کمک بیشتر، در خدمت هستم.",
                "اگر درباره این محصول یا محصولات مشابه سوالی دارید، بفرمایید.",
            ])
            dialogue.append({'role': 'assistant', 'text': closing_template2})
        
        elif num_rounds == 6:
            # 6 rounds: 2 features
            num_features = 2
            features_to_discuss = feature_order[:min(2, len(feature_order))]
            
            # Feature 1
            feature1_name = features_to_discuss[0]
            feature1_value = sampled_features.get(feature1_name, '')
            feature1_display = feature1_name.split('_', 1)[-1] if '_' in feature1_name else feature1_name
            
            # Round 2: Assistant asks about feature 1
            assistant_template = random.choice(self.assistant_search_response_templates)
            assistant_question = assistant_template.format(
                title=title,
                brand=brand,
                category=category,
                feature=feature1_display
            )
            dialogue.append({
                'role': 'assistant',
                'text': assistant_question
            })
            
            # Round 3: User answers feature 1
            user_answer_template = random.choice(self.user_answer_templates)
            user_answer = user_answer_template.format(
                feature=feature1_display,
                value=feature1_value
            )
            dialogue.append({
                'role': 'user',
                'text': user_answer
            })
            
            # Feature 2
            if len(features_to_discuss) > 1:
                feature2_name = features_to_discuss[1]
                feature2_value = sampled_features.get(feature2_name, '')
                feature2_display = feature2_name.split('_', 1)[-1] if '_' in feature2_name else feature2_name
                
                # Round 4: Assistant asks about feature 2
                assistant_template2 = random.choice(self.assistant_feature_questions)
                assistant_question2 = assistant_template2.format(
                    feature=feature2_display,
                    title=title
                )
                dialogue.append({
                    'role': 'assistant',
                    'text': assistant_question2
                })
                
                # Round 5: User answers feature 2
                user_answer_template2 = random.choice(self.user_answer_templates)
                user_answer2 = user_answer_template2.format(
                    feature=feature2_display,
                    value=feature2_value
                )
                dialogue.append({
                    'role': 'user',
                    'text': user_answer2
                })
            
            # Round 6: Assistant closing with product recommendation
            closing_template = random.choice(self.closing_templates)
            closing = closing_template.format(
                title=title,
                brand=brand
            )
            dialogue.append({
                'role': 'assistant',
                'text': closing
            })
        
        else:  # 7 rounds
            # 7 rounds: 2 features with confirmation
            features_to_discuss = feature_order[:min(2, len(feature_order))]
            
            # Feature 1
            feature1_name = features_to_discuss[0]
            feature1_value = sampled_features.get(feature1_name, '')
            feature1_display = feature1_name.split('_', 1)[-1] if '_' in feature1_name else feature1_name
            
            # Round 2: Assistant asks about feature 1
            assistant_template = random.choice(self.assistant_search_response_templates)
            assistant_question = assistant_template.format(
                title=title,
                brand=brand,
                category=category,
                feature=feature1_display
            )
            dialogue.append({
                'role': 'assistant',
                'text': assistant_question
            })
            
            # Round 3: User answers feature 1
            user_answer_template = random.choice(self.user_answer_templates)
            user_answer = user_answer_template.format(
                feature=feature1_display,
                value=feature1_value
            )
            dialogue.append({
                'role': 'user',
                'text': user_answer
            })
            
            # Round 4: Assistant confirms and asks about feature 2
            if len(features_to_discuss) > 1:
                feature2_name = features_to_discuss[1]
                feature2_value = sampled_features.get(feature2_name, '')
                feature2_display = feature2_name.split('_', 1)[-1] if '_' in feature2_name else feature2_name
                
                follow_up_template = random.choice(self.assistant_follow_up_templates)
                assistant_follow_up = follow_up_template.format(
                    feature=feature1_display,
                    value=feature1_value,
                    next_feature=feature2_display
                )
                dialogue.append({
                    'role': 'assistant',
                    'text': assistant_follow_up
                })
                
                # Round 5: User answers feature 2
                user_answer_template2 = random.choice(self.user_answer_templates)
                user_answer2 = user_answer_template2.format(
                    feature=feature2_display,
                    value=feature2_value
                )
                dialogue.append({
                    'role': 'user',
                    'text': user_answer2
                })
                
                # Round 6: Assistant confirms feature 2
                confirmation_template = random.choice(self.assistant_confirmation_templates)
                assistant_confirmation = confirmation_template.format(
                    feature=feature2_display,
                    value=feature2_value
                )
                dialogue.append({
                    'role': 'assistant',
                    'text': assistant_confirmation
                })
            else:
                # Only one feature available
                confirmation_template = random.choice(self.assistant_confirmation_templates)
                assistant_confirmation = confirmation_template.format(
                    feature=feature1_display,
                    value=feature1_value
                )
                dialogue.append({
                    'role': 'assistant',
                    'text': assistant_confirmation
                })
            
            # Round 7: Assistant closing with product recommendation
            closing_template = random.choice(self.closing_templates)
            closing = closing_template.format(
                title=title,
                brand=brand
            )
            dialogue.append({
                'role': 'assistant',
                'text': closing
            })
        
        # Add metadata
        return {
            'product_id': product.get('id'),
            'product_title': title,
            'brand': brand,
            'category': category,
            'cluster_id': sample.get('cluster_id'),
            'dialogue_rounds': len(dialogue),
            'features_discussed': features_to_discuss,
            'dialogue': dialogue
        }
    
    def generate_dialogues_batch(self, num_dialogues: int = None, num_rounds: int = 14):
        """Generate dialogues for multiple products"""
        if num_dialogues is None:
            num_dialogues = len(self.samples)
        
        print(f"\nGenerating {num_dialogues} dialogues...")
        dialogues = []
        
        valid_samples = [s for s in self.samples if 'error' not in s and s.get('sampled_features')]
        
        for i, sample in enumerate(valid_samples[:num_dialogues]):
            if i % 50 == 0:
                print(f"  Generated {i}/{num_dialogues} dialogues...")
            
            dialogue = self.generate_dialogue(sample, num_rounds=num_rounds)
            if dialogue:
                dialogues.append(dialogue)
        
        self.dialogues = dialogues
        print(f"\nGenerated {len(dialogues)} dialogues")
        return dialogues
    
    def save_dialogues(self, output_file: str):
        """Save generated dialogues to JSON file"""
        output_data = {
            'metadata': {
                'total_dialogues': len(self.dialogues),
                'generation_date': str(__import__('datetime').datetime.now()),
            },
            'dialogues': self.dialogues
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(self.dialogues)} dialogues to {output_file}")
    
    def print_sample_dialogue(self, index: int = 0):
        """Print a sample dialogue for preview"""
        if not self.dialogues or index >= len(self.dialogues):
            print("No dialogues available")
            return
        
        dialogue_data = self.dialogues[index]
        print("\n" + "=" * 70)
        print(f"Sample Dialogue {index + 1}")
        print("=" * 70)
        print(f"\nProduct: {dialogue_data['product_title']}")
        print(f"Brand: {dialogue_data['brand']}")
        print(f"Category: {dialogue_data['category']}")
        print(f"Rounds: {dialogue_data['dialogue_rounds']}")
        print(f"\nDialogue:")
        print("-" * 70)
        
        for i, turn in enumerate(dialogue_data['dialogue'], 1):
            role = turn['role']
            text = turn['text']
            role_icon = "👤" if role == 'user' else "🤖"
            print(f"\n{role_icon} [{role.upper()} - Round {i}]:")
            print(f"   {text}")
        
        print("\n" + "=" * 70)

def main():
    """Main function to generate dialogues"""
    print("=" * 70)
    print("DIALOGUE TEMPLATE GENERATOR")
    print("=" * 70)
    
    # Configuration
    sampled_products_file = "sampled_products/batch_500_products_5_features.json"
    output_file = "generated_dialogues/dialogues_template.json"
    
    # Initialize generator
    generator = DialogueGenerator(sampled_products_file)
    
    try:
        # Load samples
        generator.load_samples()
        
        # Generate dialogues (first 100 for testing, or all if you want)
        num_dialogues = 500  # Change to None to generate for all products
        generator.generate_dialogues_batch(num_dialogues=num_dialogues)
        
        # Save dialogues
        import os
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        generator.save_dialogues(output_file)
        
        # Print sample dialogues
        print("\n" + "=" * 70)
        print("SAMPLE DIALOGUES")
        print("=" * 70)
        
        for i in range(min(3, len(generator.dialogues))):
            generator.print_sample_dialogue(i)
        
        print("\n" + "=" * 70)
        print("DIALOGUE GENERATION COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Dialogues saved to: {output_file}")
        print(f"📊 Total dialogues: {len(generator.dialogues)}")
        print(f"💬 Each dialogue has 14 rounds (5 features)")
        
    except Exception as e:
        print(f"Error during dialogue generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
