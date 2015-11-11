#include <vector>

template <typename T>
class BTreeNode {
private:
	BTreeNode<T>* _parent;

	std::vector<T> _values;
	std::vector<BTreeNode*> _children;

	BTreeNode(std::vector<T> initialValues) {
		_values = initialValues;
		_parent = NULL;
	};

	BTreeNode(std::vector<T> initialValues, std::vector<BTreeNode*> initialChildren) {
		_values = initialValues;
		_children = initialChildren;
		_parent = NULL;
	};

	void insert_up(T value, BTreeNode* rightChild, unsigned int nodeSize) {
		std::vector<T>::iterator valueIterator;
		std::vector<BTreeNode*>::iterator childIterator;
		valueIterator = _values.begin();
		childIterator = _children.begin();
		while (valueIterator != _values.end() && *valueIterator < value) {
			valueIterator++;
			childIterator++;
		}

		// Biggest value; insert at end
		if (valueIterator == _values.end()) {
			_values.push_back(value);
			_children.push_back(rightChild);
			return;
		}

		_values.insert(valueIterator, value);
		childIterator++;
		_children.insert(childIterator, rightChild);

		if (_values.size() > nodeSize) {
			split(nodeSize);
		}
	};

	void set_parent(BTreeNode<T>* parent) {
		_parent = parent;
	}

	void split(unsigned int nodeSize) {
		unsigned int medianIndex = _values.size() / 2;
		T median = _values[medianIndex];
		// Take the values smaller and bigger than the median for the new children
		std::vector<T> leftValues(_values.begin(), _values.begin() + medianIndex);
		std::vector<T> rightValues(_values.begin() + medianIndex + 1, _values.end());

		std::vector<BTreeNode*> leftChildren, rightChildren;
		// Take the children linked to the above values for the new children
		if (_children.size() > 0) {
			leftChildren = std::vector<BTreeNode*>(_children.begin(), _children.begin() + medianIndex + 1);
			rightChildren = std::vector<BTreeNode*>(_children.begin() + medianIndex + 1, _children.end());
		}

		// This node becomes the left child
		_values = leftValues;
		_children = leftChildren;

		BTreeNode* rightNode = new BTreeNode(rightValues, rightChildren);

		if (!_parent) {
			// This node is the current root node so we create a new parent
			std::vector<T> parentValues;
			parentValues.push_back(median);
			std::vector<BTreeNode*> parentChildren;
			parentChildren.push_back(this);
			parentChildren.push_back(rightNode);

			_parent = new BTreeNode(parentValues, parentChildren);
			rightNode->set_parent(_parent);
		}
		else {
			rightNode->set_parent(_parent);
			// Insert the median in the parent node
			_parent->insert_up(median, rightNode, nodeSize);
		}
	}

public:
	BTreeNode() {
		_parent = NULL;
	};

	void insert(T value, unsigned int nodeSize) {
		// Leaf node
		if (_children.size() == 0) {
			std::vector<T>::iterator valueIterator = _values.begin();
			while (valueIterator != _values.end() && *valueIterator < value) {
				valueIterator++;
			}
			_values.insert(valueIterator, value);

			if (_values.size() <= nodeSize) {
				return;
			}
			split(nodeSize);
			return;
		}

		unsigned int i = 0;
		while (i < _values.size() && _values[i] < value) {
			i++;
		}

		_children[i]->insert(value, nodeSize);
	};

	bool contains(T value) {
		unsigned int i = 0;
		while (i < _values.size() && _values[i] < value) {
			i++;
		}

		// Value bigger than any in this node
		if (i == _values.size()) {
			if (_children.size() > 0) {
				return _children[i]->contains(value);
			}
			else {
				return false;
			}
		}

		// We found the value
		if (_values[i] == value) {
			return true;
		}

		if (_children.size() > 0) {
			// Continue search in appropriate child node
			return _children[i]->contains(value);
		}
		else {
			// This is a leaf and it does not contain the value
			return false;
		}
	};

	int size() {
		int size = _values.size();
		for each (BTreeNode* child in _children) {
			size += child->size();
		}
		return size;
	};

	std::string print(int depth) {
		std::string depthString = "(" + std::to_string(depth) + ");";
		std::string result = "{";
		if (_children.size() == 0) {
			for each (T element in _values) {
				result += std::to_string(element) + depthString;
			}
			return result + "}";
		}
		for (unsigned int i = 0; i < _values.size(); i++) {
			result += _children[i]->print(depth + 1);
			result += std::to_string(_values[i]) + depthString;
		}
		result += _children[_children.size() - 1]->print(depth + 1);
		return result + "}";
	};

	BTreeNode<T>* get_parent() {
		return _parent;
	}
};

template <typename T>
class BTree {
private:
	BTreeNode<T>* _root;
	unsigned int _nodeSize;
public:
	BTree(unsigned int nodeSize) {
		_nodeSize = nodeSize;
		_root = new BTreeNode<T>();
	};

	void insert(T value) {
		_root->insert(value, _nodeSize);
		BTreeNode<T>* parent = _root->get_parent();
		while (parent) {
			_root = parent;
			parent = _root->get_parent();
		}
	};

	bool contains(T value) {
		return _root->contains(value);
	};
	
	int size() {
		return _root->size();
	}

	std::string print() {
		return _root->print(0);
	}
};
