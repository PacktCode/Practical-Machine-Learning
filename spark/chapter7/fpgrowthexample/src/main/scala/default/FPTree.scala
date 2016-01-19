// Practical Machine learning
// Association rule based learning - FPGrowth example
// Chapter 7

package default

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

/**
 * File: FPTree.scala
 * Description: This is an implementation of the FPTree as described in Han, Jiawei, et al.,
 *              "Mining frequent patterns without candidate generation: A frequent-pattern tree approach",
 *              data mining and knowledge discovery, 2004.
 * Author: Lin, Chen
 * E-mail: chlin.ecnu@gmail.com
 * Version: 2.4
 */

object FPTree{
  def apply(records: Array[Array[String]], minSupport: Long): ArrayBuffer[(String, Long)] = {
    //Transform Array[Array[String]] to ArrayBuffer[ArrayBuffer[String]] in case elements of transactions will be modified.
    var transactions = new ArrayBuffer[ArrayBuffer[String]]()
    for(record <- records){
      var transaction = new ArrayBuffer[String]()
      record.copyToBuffer(transaction)
      transactions += transaction
    }

    //Create FPTree.
    val fptree = new FPTree(new ArrayBuffer[(String, Long)]())

    //Run FPGrowth.
    fptree.FPGrowth(transactions, new ArrayBuffer[String](), minSupport)

    //Return frequent patterns mined from fptree.
    fptree.patterns
  }//End of apply method
} //End of object FPTree.

class FPTree(var patterns: ArrayBuffer[(String, Long)]){

  /**
   *
   * @param transactions
   * @param minSupport
   * @return
   */
  def buildHeaderTable(transactions: ArrayBuffer[ArrayBuffer[String]], minSupport: Long): ArrayBuffer[TreeNode] = {
    if(transactions.nonEmpty){
      val map: HashMap[String, TreeNode] = new HashMap[String, TreeNode]()
      for(transaction <- transactions){
        for(name <- transaction){
          if(!map.contains(name)){
            val node: TreeNode = new TreeNode(name)
            node.count = 1
            map(name) = node
          }else{
            map(name).count += 1
          }
        }
      }
      val headerTable = new ArrayBuffer[TreeNode]()
      map.filter(_._2.count >= minSupport).values.toArray.sortWith(_.name < _.name).sortWith(_.count > _.count).copyToBuffer(headerTable)
      headerTable //return headerTable
    }else{
      null //if transactions is empty, return null
    }
  } //end of buildHeaderTable

  def buildFPTree(transactions: ArrayBuffer[ArrayBuffer[String]], headerTable: ArrayBuffer[TreeNode]): TreeNode = {
    val root: TreeNode = new TreeNode()
    for(transaction <- transactions){
      val sortedTransaction = sortByHeaderTable(transaction, headerTable)
      var subTreeRoot: TreeNode = root
      var tmpRoot: TreeNode = null
      if(root.children.nonEmpty){
        while(sortedTransaction.nonEmpty && subTreeRoot.findChild(sortedTransaction.head) != null){
          tmpRoot = subTreeRoot.children.find(_.name.equals(sortedTransaction.head)) match {
            case Some(node) => node
            case None => null
          }
          tmpRoot.count += 1
          subTreeRoot = tmpRoot
          sortedTransaction.remove(0)
        } //end of while
      } //end of if
      addNodes(subTreeRoot, sortedTransaction, headerTable)
    } //end of for

    def sortByHeaderTable(transaction: ArrayBuffer[String], headerTable: ArrayBuffer[TreeNode]): ArrayBuffer[String] = {
      val map: HashMap[String, Long] = new HashMap[String, Long]()
      for(item <- transaction){
        for(index <- 0 until headerTable.length){
          if(headerTable(index).name.equals(item)){
            map(item) = index
          }
        }
      }

      val sortedTransaction: ArrayBuffer[String] = new ArrayBuffer[String]()
      map.toArray.sortWith(_._2 < _._2).foreach(sortedTransaction += _._1)
      sortedTransaction //return sortedTransaction
    } //end of sortByHeaderTable

    def addNodes(parent: TreeNode, transaction: ArrayBuffer[String], headerTable: ArrayBuffer[TreeNode]){
      while(transaction.nonEmpty){
        val name: String = transaction.head
        transaction.remove(0)
        val leaf: TreeNode = new TreeNode(name)
        leaf.count = 1
        leaf.parent = parent
        parent.children += leaf

        var break = true //for breaking out of while loop
        var index: Int = 0
        while(break && index < headerTable.length){
          var node = headerTable(index)
          if(node.name.equals(name)){
            while(node.nextHomonym != null)
              node = node.nextHomonym
            node.nextHomonym  = leaf
            break = false
          }
          index += 1
        }

        addNodes(leaf, transaction, headerTable)
      }
    } //end of addNodes

    root //return root
  } //end of buildFPTree

  def FPGrowth(transactions: ArrayBuffer[ArrayBuffer[String]], postPattern: ArrayBuffer[String], minSupport: Long){
    val headerTable: ArrayBuffer[TreeNode] = buildHeaderTable(transactions, minSupport)

    val treeRoot = buildFPTree(transactions, headerTable)

    if(treeRoot.children.nonEmpty){
      if(postPattern.nonEmpty){
        for(node <- headerTable){
          var result: String = ""
          val temp = new ArrayBuffer[String]()
          temp += node.name
          for(pattern <- postPattern){
            temp += pattern
          }
          result += temp.sortWith(_ < _).mkString(" ").toString
          patterns += result -> node.count
        }
      }

    for (node: TreeNode <- headerTable) {
      val newPostPattern: ArrayBuffer[String] = new ArrayBuffer[String]()
      newPostPattern += node.name
      if (postPattern.nonEmpty)
        newPostPattern ++= postPattern
      val newTransactions: ArrayBuffer[ArrayBuffer[String]] = new ArrayBuffer[ArrayBuffer[String]]()
      var backNode: TreeNode = node.nextHomonym
      while (backNode != null) {
        var counter: Long = backNode.count
        val preNodes: ArrayBuffer[String] = new ArrayBuffer[String]()
        var parent: TreeNode = backNode.parent
        while (parent.name != null) {
          preNodes += parent.name
          parent = parent.parent
        }
        while (counter > 0) {
          newTransactions += preNodes
          counter -= 1
        }
        backNode = backNode.nextHomonym
      }

      FPGrowth(newTransactions, newPostPattern, minSupport)
    } //end of for

    }
  } //end of FPGrowth
} //end of FPTree


