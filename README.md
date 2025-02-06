# Configurable Components Library

## Installation

To install this package, you can use **Conda** with the included development tools:

```bash
conda env create -f environment.yml
```

## Usage

## Key Concepts

This package allows adding or modifying any component in a modular way thanks to the architecture based on
`Customizable` and configuration schemas. All components (models, datasets, optimizers, metrics, etc.) follow this principle.

---

### Modular Architecture with Customizable and TypedCustomizable

This library relies on a modular architecture through the base classes `Customizable` and `TypedCustomizable`. These
classes provide flexible, extensible, and standardized configuration of components (models, datasets, optimizers, etc.).

#### 1. **Customizable**: Dynamic Component Creation

`Customizable` is a base class that uses schemas (`Schema`) to dynamically validate configurations.
It enables:

- **Validation**: Each parameter is validated by type and constraint before instantiation using the `Schema` class.
- **Flexibility**: Loading configurations from Python dictionaries or YAML files. The configurations are dynamic since the parameters depend on the requested object/class type.
# Configurable Components Library

## Installation

To install this package, you can use **Conda** with the included development tools:

```bash
conda env create -f environment.yml
```

## Usage

## Key Concepts

This package allows adding or modifying any component in a modular way thanks to the architecture based on
`Customizable` and configuration schemas. All components (models, datasets, optimizers, metrics, etc.) follow this principle.

---

### Modular Architecture with Customizable and TypedCustomizable

This library relies on a modular architecture through the base classes `Customizable` and `TypedCustomizable`. These
classes provide flexible, extensible, and standardized configuration of components (models, datasets, optimizers, etc.).

#### 1. **Customizable**: Dynamic Component Creation

`Customizable` is a base class that uses schemas (`Schema`) to dynamically validate configurations.
It enables:

- **Validation**: Each parameter is validated by type and constraint before instantiation using the `Schema` class.
- **Flexibility**: Loading configurations from Python dictionaries or YAML files. The configurations are dynamic since the parameters depend on the requested object/class type.
- **Automatic attribute assignment**: Configuration parameters are automatically set as instance attributes, removing the need to manually assign them in the `__init__` method.
- **Automatic precondition checks**: The `preconditions()` method is automatically called, ensuring validation before instantiation.

**Example**:

```python
from configs.config import Customizable, Schema

class MyComponent(Customizable):
  config_schema = {
    'learning_rate': Schema(float, default=0.01),
    'batch_size': Schema(int, default=32),
  }

  def preconditions(self):
      assert self.learning_rate > 0, "Learning rate must be positive"

  def __init__(self):
      pass
```

#### 2. **TypedCustomizable**: Dynamic Subclass Management with Abstraction

`TypedCustomizable` extends `Customizable` by adding the ability to dynamically select a subclass to instantiate based on a `type` parameter.

To ensure proper implementation, abstract base classes (`ABC`) can be used to enforce method definitions in subclasses.

**Example: Using TypedCustomizable for Automatic Component Selection with Abstract Methods**

```python
from configs.config import TypedCustomizable, Schema
import abc

class BaseComponent(TypedCustomizable, abc.ABC):
  aliases = ['base_component']

  @abc.abstractmethod
  def process(self):
      """This method must be implemented in subclasses"""
      pass

class SpecificComponentA(BaseComponent):
  aliases = ['component_a']
  config_schema = {
    'param1': Schema(int, default=10),
  }

  def process(self):
      return f"Processing with param1: {self.param1}"

class SpecificComponentB(BaseComponent):
  aliases = ['component_b']
  config_schema = {
    'param2': Schema(str, default="default_value"),
  }

  def process(self):
      return f"Processing with param2: {self.param2}"

config_a = {'type': 'component_a', 'param1': 20}
component_a = BaseComponent.from_config(config_a)
print(component_a.process())  # Processing with param1: 20

config_b = {'type': 'component_b', 'param2': "custom_value"}
component_b = BaseComponent.from_config(config_b)
print(component_b.process())  # Processing with param2: custom_value
```

### Why Use This Library?

By leveraging `Customizable` and `TypedCustomizable`, this library allows:

- **Modular and scalable design**: New components can be added with minimal modifications.
- **Configuration-driven instantiation**: Easily switch between different implementations using YAML or JSON configurations.
- **Strong type and schema validation**: Ensures correct parameters and prevents misconfigurations.
- **Abstract base classes for contract enforcement**: Guarantees that all subclasses implement required methods.
- **Preconditions to validate component state**: Ensures that instantiated components are correctly configured without requiring manual calls.

### Schema Functionality

#### `Schema` Class Concept

The `Schema` class defines the expected structure for each configuration parameter. It plays a central role in validation and default value application during object instantiation.

Main attributes of `Schema`:

- `type`: Specifies the expected type (e.g., int, float, str).
- `default`: Defines a default value if the parameter is not provided.
- `optional`: Indicates whether the parameter is optional.
- `aliases`: Allows using alternative names for the same parameter.

A predefined `Config` type is also provided for flexibility:

```python
from typing import Union
Config = Union[dict, str]
```

This allows configuration data to be passed as either a dictionary or a YAML file path.

### Adding a Component in Practice

#### `Customizable`

Define the class: Inherit from the appropriate base class (e.g., `BaseComponent`) or directly from `Customizable` and implement the required logic.

```python
class NewComponent(Customizable):
  config_schema = {
    'param1': Schema(str),
    'param2': Schema(int, default=10),
  }

  def preconditions(self):
      assert self.param2 >= 0, "param2 must be non-negative"
```

Using configuration-based instantiation:

```yaml
component:
  param1: "example"
```

```python
import NewComponent

component = NewComponent.from_config(config['component'])
```

With `TypedCustomizable`, dynamically selecting the right implementation is straightforward, making this approach ideal for large-scale, evolving systems.


Contact: julienrabault@icloud.com
