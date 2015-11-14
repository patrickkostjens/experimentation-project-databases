#include <vector>

template <typename Key, typename Value>
struct IndexItem {
	Key key;
	Value value;

	IndexItem(Key k, Value v) {
		key = k;
		value = v;
	};
};

template <typename Key, typename Value>
class BTreeNode {
private:
	BTreeNode* _parent;

	std::vector<IndexItem<Key, Value>> _values;
	std::vector<BTreeNode*> _children;

	BTreeNode(std::vector<IndexItem<Key, Value>> initialValues) {
		_values = initialValues;
		_parent = NULL;
	};

	BTreeNode(std::vector<IndexItem<Key, Value>> initialValues, std::vector<BTreeNode*> initialChildren) {
		_values = initialValues;
		_children = initialChildren;
		_parent = NULL;
	};

	void insert_up(IndexItem<Key, Value> element, BTreeNode* rightChild, unsigned int nodeSize) {
		std::vector<IndexItem<Key, Value>>::iterator valueIterator;
		std::vector<BTreeNode*>::iterator childIterator;
		valueIterator = _values.begin();
		childIterator = _children.begin();
		while (valueIterator != _values.end() && valueIterator->key < element.key) {
			valueIterator++;
			childIterator++;
		}

		// Biggest value; insert at end
		if (valueIterator == _values.end()) {
			_values.push_back(element);
			_children.push_back(rightChild);
			return;
		}

		_values.insert(valueIterator, element);
		childIterator++;
		_children.insert(childIterator, rightChild);

		if (_values.size() > nodeSize) {
			split(nodeSize);
		}
	};

	void set_parent(BTreeNode* parent) {
		_parent = parent;
	}

	void split(unsigned int nodeSize) {
		unsigned int medianIndex = _values.size() / 2;
		IndexItem<Key, Value> median = _values[medianIndex];
		// Take the values smaller and bigger than the median for the new children
		std::vector<IndexItem<Key, Value>> leftValues(_values.begin(), _values.begin() + medianIndex);
		std::vector<IndexItem<Key, Value>> rightValues(_values.begin() + medianIndex + 1, _values.end());

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
			std::vector<IndexItem<Key, Value>> parentValues;
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

	void insert(Key key, Value value, unsigned int nodeSize) {
		// Leaf node
		if (_children.size() == 0) {
			std::vector<IndexItem<Key, Value>>::iterator valueIterator = _values.begin();
			while (valueIterator != _values.end() && valueIterator->key < key) {
				valueIterator++;
			}
			_values.insert(valueIterator, IndexItem<Key, Value>(key, value));

			if (_values.size() <= nodeSize) {
				return;
			}
			split(nodeSize);
			return;
		}

		unsigned int i = 0;
		while (i < _values.size() && _values[i].key < key) {
			i++;
		}

		_children[i]->insert(key, value, nodeSize);
	};

	bool contains(Key key) {
		unsigned int i = 0;
		while (i < _values.size() && _values[i].key < key) {
			i++;
		}

		// Value bigger than any in this node
		if (i == _values.size()) {
			if (_children.size() > 0) {
				return _children[i]->contains(key);
			}
			else {
				return false;
			}
		}

		// We found the value
		if (_values[i].key == key) {
			return true;
		}

		if (_children.size() > 0) {
			// Continue search in appropriate child node
			return _children[i]->contains(key);
		}
		else {
			// This is a leaf and it does not contain the value
			return false;
		}
	};

	void find(Key key, std::vector<Value>* result) {
		unsigned int i = 0;
		while (i < _values.size() && _values[i].key < key) {
			i++;
		}

		// Value bigger than any in this node
		if (i == _values.size()) {
			if (_children.size() > 0) {
				_children[i]->find(key, result);
				return;
			}
			else {
				return;
			}
		}

		// We found the value; search this node and appropriate child nodes for all occurrences
		if (_values[i].key == key) {
			while (i < _values.size() && _values[i].key == key) {
				result->push_back(_values[i].value);
				if (_children.size() > 0) {
					_children[i]->find(key, result);
				}
				i++;
			}
			if (i < _children.size()) {
				_children[i]->find(key, result);
			}
			return;
		}

		if (_children.size() > 0) {
			// Continue search in appropriate child node
			_children[i]->find(key, result);
			return;
		}
		else {
			// This is a leaf and it does not contain the value
			return;
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
			for each (IndexItem<Key, Value> element in _values) {
				result += std::to_string(element.key) + depthString;
			}
			return result + "}";
		}
		for (unsigned int i = 0; i < _values.size(); i++) {
			result += _children[i]->print(depth + 1);
			result += std::to_string(_values[i].key) + depthString;
		}
		result += _children[_children.size() - 1]->print(depth + 1);
		return result + "}";
	};

	BTreeNode* get_parent() {
		return _parent;
	};

	~BTreeNode() {
		for (std::vector<BTreeNode*>::iterator it = _children.begin(); it != _children.end(); ++it)
		{
			delete (*it);
		}
	};
};

template <typename Key, typename Value>
class BTree {
private:
	BTreeNode<Key, Value>* _root;
	unsigned int _nodeSize;
public:
	BTree(unsigned int nodeSize) {
		_nodeSize = nodeSize;
		_root = new BTreeNode<Key, Value>();
	};

	void insert(Key key, Value value) {
		_root->insert(key, value, _nodeSize);
		BTreeNode<Key, Value>* parent = _root->get_parent();
		while (parent) {
			_root = parent;
			parent = _root->get_parent();
		}
	};

	bool contains(Key key) {
		return _root->contains(key);
	};
	
	std::vector<Value> find(Key key) {
		std::vector<Value> results;
		_root->find(key, &results);
		return results;
	};

	int size() {
		return _root->size();
	};

	std::string print() {
		return _root->print(0);
	};

	~BTree() {
		delete _root;
	};
};
