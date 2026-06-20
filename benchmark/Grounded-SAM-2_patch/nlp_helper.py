from nltk.corpus import wordnet as wn
import nltk


class EnhancedPersonDetector:
    def __init__(self):
        nltk.download('wordnet', quiet=True)
        self.person_keywords = {
            'person', 'human', 'individual', 'people', 'someone', 'somebody',
            'man', 'woman', 'boy', 'girl', 'child', 'baby', 'adult', 'kid',
            'male', 'female', 'gentleman', 'lady', 'teen', 'teenager',
            'doctor', 'teacher', 'lawyer', 'engineer', 'artist', 'musician',
            'actor', 'singer', 'dancer', 'writer', 'author', 'painter',
            'scientist', 'researcher', 'professor', 'student', 'worker',
            'employee', 'manager', 'director', 'officer', 'soldier',
            'nurse', 'driver', 'pilot', 'chef', 'cook', 'waiter', 'waitress',
            'player', 'athlete', 'runner', 'swimmer', 'footballer',
            'parent', 'father', 'mother', 'brother', 'sister', 'friend',
            'neighbor', 'colleague', 'partner', 'spouse', 'husband', 'wife',
            'customer', 'client', 'patient', 'visitor', 'guest',
            'hero', 'villain', 'leader', 'follower', 'expert', 'beginner',
            'professional', 'amateur', 'volunteer', 'tourist', 'traveler'
        }
        
        # 黑名单：绝对不是人
        self.blacklist = {
            # 食物
            'cookie', 'cake', 'pie', 'bread', 'biscuit', 'cracker', 'muffin',
            'cupcake', 'donut', 'bagel', 'sandwich', 'pizza', 'burger',
            # 动物
            'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'chicken',
            # 物品
            'box', 'bag', 'bottle', 'cup', 'plate', 'bowl', 'chair', 'table',
            'car', 'truck', 'bike', 'phone', 'computer',
            # 植物
            'tree', 'flower', 'plant', 'grass', 'bush'
        }
    
    def is_person(self, word, strict_mode=True):
        """
        判断是否为人
        
        Args:
            strict_mode: True=严格模式（减少误判），False=宽松模式
        """
        word_lower = word.lower()
        
        # 黑名单检查
        if word_lower in self.blacklist:
            return False
        
        # 白名单检查
        if word_lower in self.person_keywords:
            return True
        
        # 复合词检查
        if any(kw in word_lower for kw in ['man', 'woman', 'person', 'boy', 'girl']):
            # 但排除 policeman's hat 这种所有格
            if not word_lower.endswith("'s"):
                return True
        
        # WordNet检查
        return self._wordnet_check(word_lower, strict=strict_mode)
    
    def _wordnet_check(self, word, strict=True):
        """WordNet检查"""
        synsets = wn.synsets(word, pos=wn.NOUN)
        
        if not synsets:
            return False
        
        # 获取第一个（最常用）synset
        primary_synset = synsets[0]
        
        # 检查第一个synset的lexname（词汇类别）
        lexname = primary_synset.lexname()
        
        # 如果主要类别是食物、动物等，直接返回False
        if lexname in ['noun.food', 'noun.animal', 'noun.artifact', 'noun.plant']:
            return False
        
        # 如果主要类别是人，返回True
        if lexname == 'noun.person':
            return True
        
        # 在严格模式下，只检查第一个synset
        synsets_to_check = [primary_synset] if strict else synsets
        
        for synset in synsets_to_check:
            definition = synset.definition().lower()
            
            # 强排除
            if any(kw in definition for kw in ['food', 'edible', 'eaten', 'animal', 'beast']):
                continue
            
            # 检查人类关键词
            if any(kw in definition for kw in ['person', 'human', 'people', 'individual', 'someone']):
                return True
            
            # 检查上位词
            for path in synset.hypernym_paths():
                for hypernym in path[:4]:  # 只看前4层
                    name = hypernym.name().lower()
                    if 'person' in name or 'human' in name:
                        return True
        
        return False


# 完整测试
if __name__ == "__main__":
    detector = EnhancedPersonDetector()

    test_cases = [
        ('man', True),
        ('woman', True),
        ('doctor', True),
        ('musician', True),
        ('soldier', True),
        ('player', True),
        ('athlete', True),
        ('cookie', False),  # 重点测试
        ('cake', False),
        ('dog', False),
        ('cat', False),
        ('car', False),
        ('table', False),
        ('artist', True),
        ('chef', True),
        ('cook', True),  # 这个比较微妙
        ('grapefruit', False),  # 植物
    ]

    print("Word\t\tExpected\tActual\t\tResult")
    print("-" * 60)

    correct = 0
    total = len(test_cases)

    for word, expected in test_cases:
        actual = detector.is_person(word, strict_mode=True)
        result = "✓" if actual == expected else "✗"
        correct += (actual == expected)
        print(f"{word:15s}\t{expected}\t\t{actual}\t\t{result}")

    print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")
