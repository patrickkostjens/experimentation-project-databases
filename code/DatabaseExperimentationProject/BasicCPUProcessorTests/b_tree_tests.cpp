#include "stdafx.h"
#include "CppUnitTest.h"
#include "../BasicCPUProcessor/b_tree.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace BasicCPUProcessorTests
{		
	TEST_CLASS(BTreeTests) {
	private:
		template <typename T>
		bool vector_contains(std::vector<T> vector, T value) {
			return std::find(vector.begin(), vector.end(), value) != vector.end();
		};

	public:
		
		TEST_METHOD(DefaultTreeEmpty) {
			BTree<int, int> tree(3);
			Assert::AreEqual(0, tree.size());
			Assert::AreEqual(std::string("{}"), tree.print());
		};

		TEST_METHOD(TestSingleInsert) {
			BTree<int, int> tree(3);
			Assert::IsFalse(tree.contains(1));
			tree.insert(1, 1);
			Assert::AreEqual(1, tree.size());
			Assert::IsTrue(tree.contains(1));
			Assert::AreEqual(std::string("{1(0);}"), tree.print());
		};

		TEST_METHOD(TestTwoInserts) {
			BTree<int, int> tree(3);
			tree.insert(2, 2);
			tree.insert(1, 1);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::AreEqual(2, tree.size());
			Assert::AreEqual(std::string("{1(0);2(0);}"), tree.print());
		};

		TEST_METHOD(TestThreeInserts) {
			BTree<int, int> tree(3);
			tree.insert(2, 2);
			tree.insert(1, 1);
			tree.insert(3, 3);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::IsTrue(tree.contains(3));
			Assert::AreEqual(3, tree.size());
			Assert::AreEqual(std::string("{1(0);2(0);3(0);}"), tree.print());
		};

		TEST_METHOD(TestOneSplit) {
			BTree<int, int> tree(3);
			tree.insert(2, 2);
			tree.insert(1, 1);
			tree.insert(3, 3);
			tree.insert(4, 4);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::IsTrue(tree.contains(3));
			Assert::IsTrue(tree.contains(4));
			Assert::AreEqual(4, tree.size());
			Assert::AreEqual(std::string("{{1(1);2(1);}3(0);{4(1);}}"), tree.print());
		};

		TEST_METHOD(TestPushBiggestMedianUp) {
			BTree<int, int> tree(3);
			tree.insert(2, 2);
			tree.insert(1, 1);
			tree.insert(3, 3);
			tree.insert(4, 4);
			tree.insert(5, 5);
			tree.insert(6, 6);
			tree.insert(7, 7);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::IsTrue(tree.contains(3));
			Assert::IsTrue(tree.contains(4));
			Assert::IsTrue(tree.contains(5));
			Assert::IsTrue(tree.contains(6));
			Assert::IsTrue(tree.contains(7));
			Assert::AreEqual(7, tree.size());
			Assert::AreEqual(std::string("{{1(1);2(1);}3(0);{4(1);5(1);}6(0);{7(1);}}"), tree.print());
		};

		TEST_METHOD(TestPushSmallerMedianUp) {
			BTree<int, int> tree(3);
			tree.insert(7, 7);
			tree.insert(6, 6);
			tree.insert(5, 5);
			tree.insert(4, 4);
			tree.insert(3, 3);
			tree.insert(2, 2);
			tree.insert(1, 1);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::IsTrue(tree.contains(3));
			Assert::IsTrue(tree.contains(4));
			Assert::IsTrue(tree.contains(5));
			Assert::IsTrue(tree.contains(6));
			Assert::IsTrue(tree.contains(7));
			Assert::AreEqual(7, tree.size());
			Assert::AreEqual(std::string("{{1(1);2(1);3(1);}4(0);{5(1);}6(0);{7(1);}}"), tree.print());
		};

		TEST_METHOD(TestDoublePushUp) {
			BTree<int, int> tree(3);
			for (int i = 7; i >= -2; i--) {
				tree.insert(i, i);
			}
			Assert::AreEqual(10, tree.size());
			Assert::AreEqual(std::string("{{{-2(2);-1(2);}0(1);{1(2);}2(1);{3(2);}}4(0);{{5(2);}6(1);{7(2);}}}"), tree.print());
		};

		TEST_METHOD(TestDuplicateValues) {
			BTree<int, int> tree(3);
			tree.insert(1, 1);
			tree.insert(1, 1);
			tree.insert(2, 2);
			tree.insert(2, 2);
			Assert::AreEqual(4, tree.size());
			Assert::AreEqual(std::string("{{1(1);1(1);}2(0);{2(1);}}"), tree.print());
		};

		TEST_METHOD(TestNoResultsFound) {
			BTree<int, int> tree(3);
			std::vector<int> result = tree.find(1);
			unsigned int expected = 0;
			Assert::AreEqual(expected, result.size());
		};

		TEST_METHOD(TestFindSingleValue) {
			BTree<int, int> tree(3);
			tree.insert(1, 10);
			std::vector<int> result = tree.find(1);
			unsigned int expected = 1;
			Assert::AreEqual(expected, result.size());
			Assert::AreEqual(10, result[0]);
		};

		TEST_METHOD(TestFindTwoResults) {
			BTree<int, int> tree(3);
			tree.insert(1, 10);
			tree.insert(1, 11);
			std::vector<int> result = tree.find(1);
			unsigned int expected = 2;
			Assert::AreEqual(expected, result.size());
			Assert::IsTrue(vector_contains(result, 10));
			Assert::IsTrue(vector_contains(result, 11));
		};

		TEST_METHOD(TestInvalidResultNotFound) {
			BTree<int, int> tree(3);
			tree.insert(0, 9);
			tree.insert(1, 10);
			tree.insert(2, 8);
			tree.insert(1, 11);
			std::vector<int> result = tree.find(1);
			unsigned int expected = 2;
			Assert::AreEqual(expected, result.size());
			Assert::IsTrue(vector_contains(result, 10));
			Assert::IsTrue(vector_contains(result, 11));
		};

		TEST_METHOD(TestResultsInMultipleNodesFound) {
			BTree<int, int> tree(2);
			tree.insert(1, 9);
			tree.insert(1, 10);
			tree.insert(1, 8);
			tree.insert(2, 7);
			tree.insert(1, 11);
			tree.insert(1, 7);
			std::vector<int> result = tree.find(1);
			unsigned int expected = 5;
			Assert::AreEqual(expected, result.size());
			Assert::IsTrue(vector_contains(result, 7));
			Assert::IsTrue(vector_contains(result, 8));
			Assert::IsTrue(vector_contains(result, 9));
			Assert::IsTrue(vector_contains(result, 10));
			Assert::IsTrue(vector_contains(result, 11));
		};
	};
}
