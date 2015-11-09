#include <algorithm>
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

	void insert_up(T value, BTreeNode* rightChild) {
		// TODO: Fix another push up
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
	};

	void set_parent(BTreeNode<T>* parent) {
		_parent = parent;
	}
public:
	BTreeNode() {
		_parent = NULL;
	};

	void insert(T value, unsigned int nodeSize) {
		// Leaf node
		if (_children.size() == 0) {
			if (_values.size() < nodeSize) {
				_values.push_back(value);
				std::sort(_values.begin(), _values.end());
				return;
			}
			// Split node
			_values.push_back(value);
			std::sort(_values.begin(), _values.end());
			T median = _values[_values.size() / 2];
			std::vector<T> leftValues;
			for (unsigned int i = 0; i < _values.size() / 2; i++) {
				leftValues.push_back(_values[i]);
			}
			std::vector<T> rightValues;
			for (unsigned int i = (_values.size() / 2) + 1; i < _values.size(); i++) {
				rightValues.push_back(_values[i]);
			}

			// This node becomes the left leaf
			_values = leftValues;

			BTreeNode* rightNode = new BTreeNode(rightValues);

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
				_parent->insert_up(median, rightNode);
			}
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
