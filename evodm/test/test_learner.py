from evodm import DrugSelector, practice, hyperparameters
import pytest

@pytest.fixture
def ds():
    hp = hyperparameters()
    hp.TRAIN_INPUT = "fitness"
    ds = DrugSelector(hp = hp)
    return ds

def test_add_noise(ds):
    #step forward so there is actually something in the sensor
    ds.env.step()
    
    #check to see if the add_noise functionality is actually scrambling things up.
    bool_list = [None] * 3
    for i in range(3):
        fitness = ds.env.sensor[0]
        ds.add_noise()
        new_fitness = ds.env.sensor[0]
        bool_list[i] = new_fitness != fitness
    
    assert any(bool_list)
