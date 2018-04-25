import random
class PetShop(object):
    def __init__(self, animal_factory=None): #specify product for this factory
        self.pet_factory=animal_factory
    def show_pet(self):
        pet=self.pet_factory()
        print(pet.speak())

class Dog(object):
    def speak(self):
        return "w"
    def __str__(self):
        return 'Dog'
class Cat(object):
    def speak(self):
        return "w"
    def __str__(self):
        return 'Cat'

def random_animal():
    return random.choice([Dog,Cat])()


cat_shop=PetShop(Cat) #product -> abstract factory
cat_shop.show_pet()

shop=PetShop(random_animal)
for i in range(3):
    shop.show_pet()
