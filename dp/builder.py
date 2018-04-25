def construct_building(builder):  # pass in the product-builder to build
    builder.new_building()
    builder.build_floor()
    builder.build_size()
    return builder.building


class Builder(object):
    def __init__(self):
        self.building = None

    def new_building(self):
        self.building = Building()

    def build_floor(self):
        raise NotImplementedError

    def build_size(self):
        raise NotImplementedError


class BuilderHouse(Builder):
    def build_flooor(self):
        self.building.floor = 'One'

    def build_size(self):
        self.building.size = 'Big'


class BuilderFlat(Builder):
    def build_flooor(self):
        self.building.floor = 'One'

    def build_size(self):
        self.building.size = 'Big'


# product
class Building(object):
    def __init__(self):
        self.floor = None
        self.size = None

    def __repr__(self):
        return (self.floor + self.size)


buidling = construct_building(BuilderHouse)
print(buidling)
buidling = construct_building(BuilderFlat)
print(buidling)
