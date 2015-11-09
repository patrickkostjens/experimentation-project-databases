#include "stdafx.h"
#include "CppUnitTest.h"
#include "../BasicCPUProcessor/b_tree.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace BasicCPUProcessorTests
{		
	TEST_CLASS(BTreeTests) {
	public:
		
		TEST_METHOD(DefaultTreeEmpty) {
			BTree<int> tree(3);
			Assert::AreEqual(0, tree.size());
			Assert::AreEqual(std::string("{}"), tree.print());
		};

		TEST_METHOD(TestSingleInsert) {
			BTree<int> tree(3);
			Assert::IsFalse(tree.contains(1));
			tree.insert(1);
			Assert::AreEqual(1, tree.size());
			Assert::IsTrue(tree.contains(1));
			Assert::AreEqual(std::string("{1(0);}"), tree.print());
		};

		TEST_METHOD(TestTwoInserts) {
			BTree<int> tree(3);
			tree.insert(2);
			tree.insert(1);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::AreEqual(2, tree.size());
			Assert::AreEqual(std::string("{1(0);2(0);}"), tree.print());
		};

		TEST_METHOD(TestThreeInserts) {
			BTree<int> tree(3);
			tree.insert(2);
			tree.insert(1);
			tree.insert(3);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::IsTrue(tree.contains(3));
			Assert::AreEqual(3, tree.size());
			Assert::AreEqual(std::string("{1(0);2(0);3(0);}"), tree.print());
		};

		TEST_METHOD(TestOneSplit) {
			BTree<int> tree(3);
			tree.insert(2);
			tree.insert(1);
			tree.insert(3);
			tree.insert(4);
			Assert::IsTrue(tree.contains(1));
			Assert::IsTrue(tree.contains(2));
			Assert::IsTrue(tree.contains(3));
			Assert::IsTrue(tree.contains(4));
			Assert::AreEqual(4, tree.size());
			Assert::AreEqual(std::string("{{1(1);2(1);}3(0);{4(1);}}"), tree.print());
		};

		TEST_METHOD(TestPushBiggestMedianUp) {
			BTree<int> tree(3);
			tree.insert(2);
			tree.insert(1);
			tree.insert(3);
			tree.insert(4);
			tree.insert(5);
			tree.insert(6);
			tree.insert(7);
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
			BTree<int> tree(3);
			tree.insert(7);
			tree.insert(6);
			tree.insert(5);
			tree.insert(4);
			tree.insert(3);
			tree.insert(2);
			tree.insert(1);
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
	};
}