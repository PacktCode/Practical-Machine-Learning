// Practical Machine learning
// Association rule based learning - FPGrowth example
// Chapter 7

package default

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import scala.collection.mutable.{ArrayBuffer, HashMap, Set}

/**
 * File: ParallelFPGrowth.scala
 * Description: This is an implementation of Parallel FPGrowth as described in Haoyuan Li, et al.,
 * "PFP: Parallel FP-Growth for Query Recommendation", Proceedings of the 2008 ACM conference on recommender systems, 2008.
 * Author: Lin, Chen
 * E-mail: chlin.ecnu@gmail.com
 * Version: 2.5
 */

object ParallelFPGrowth {
  def apply(
      sc:SparkContext,
      input: RDD[String],
      minSupport: Long,
      splitterPattern: String,
      numGroups: Long): RDD[(String, Long)] = {
    /**
     * Split each line of transactions into items, flatMap transformation will transform all items of all lines to one RDD[String].
     * Then we count the number of times every item appears in the transaction database and it is the so-called support count.
     * For short, we just call it support.
     * After that, we will filter fList by minSupport and sort it by items' names and suppports.
     * This RDD[(String, Int)] need to be cached and collect.
     */

    val fList= input.flatMap(line => line.split(splitterPattern))
      .map(item => (item.toLong, 1.toLong)).reduceByKey(_ + _).filter(_._2 >= minSupport)
      .sortBy(pair => pair._2, false).collect

    // For the convenience of deviding all items in fList to numGroups groups,
    // we assign a unique ID which starts from 0 to (the length of fList - 1) to every item in fList.
    val fMap = getFMap(fList)
    val broadcastFMap = sc.broadcast(fMap)

    // Compute the number of items in each group.
    val maxPerGroup = getMaxPerGroup(fList, numGroups)
    val patterns = input.map(line => line.split(splitterPattern))
      .flatMap(line => ParallelFPGrowthMapper().map(line, broadcastFMap, maxPerGroup))
      .groupByKey.flatMap(line => ParallelFPGrowthReducer().reduce(line, minSupport, maxPerGroup))
      .groupByKey().map(line => (line._1, line._2.max))

    patterns
  }

  def getFMap(fList: Array[(Long, Long)]) : HashMap[String, Long] = {
    var i = 0
    val fMap = new HashMap[String, Long]()
    for(pair <- fList){
      fMap.put(pair._1.toString, i)
      i += 1
    }
    fMap
  }

  def getMaxPerGroup(fList: Array[(Long, Long)], numGroups: Long): Long = {
    var maxPerGroup = fList.length / numGroups
    if (fList.length % numGroups != 0) {
      maxPerGroup += 1
    }
    maxPerGroup
  }
}

object ParallelFPGrowthMapper{
  def apply(): ParallelFPGrowthMapper = {
    val mapper = new ParallelFPGrowthMapper()
    mapper
  }
}

class ParallelFPGrowthMapper(){
  def map(
      transaction: Array[String],
      bcFMap: Broadcast[HashMap[String, Long]],
      maxPerGroup: Long): ArrayBuffer[(Long, ArrayBuffer[String])] = {

    def getGroupID(itemId: Long, maxPerGroup: Long): Long ={
      itemId / maxPerGroup
    } //end of getGroupID

    var retVal = new ArrayBuffer[(Long, ArrayBuffer[String])]()
    var itemArr = new ArrayBuffer[Long]()
    val fMap = bcFMap.value
    for(item <- transaction){
      if(fMap.keySet.contains(item)){
        itemArr += fMap(item)
      }
    }
    itemArr = itemArr.sortWith(_ < _)
    val groups = new ArrayBuffer[Long]()
    for(i <- (0 until itemArr.length).reverse){
      val item = itemArr(i)
      val groupID = getGroupID(item, maxPerGroup)
      if(!groups.contains(groupID)){
        val tempItems = new ArrayBuffer[Long]()
        tempItems ++= itemArr.slice(0, i + 1)
        val items = tempItems.map(x => fMap.map(_.swap).getOrElse(x, null))
        retVal += groupID -> items
        //retVal += groupID -> tempItems
        groups += groupID
      }
    }
    retVal
  } //end of map
} //end of ParallelFPGrowthMapper

object ParallelFPGrowthReducer{
  def apply(): ParallelFPGrowthReducer ={
    val reducer = new ParallelFPGrowthReducer()
    reducer
  }
}

class ParallelFPGrowthReducer(){
  def reduce(
      line: (Long, Iterable[ArrayBuffer[String]]),
      minSupport: Long,
      maxPerGroup: Long): ArrayBuffer[(String, Long)] ={
    val transactions = line._2
    val localFPTree = new LocalFPTree(new ArrayBuffer[(String, Long)])
    localFPTree.FPGrowth(transactions, new ArrayBuffer[String], minSupport)
    localFPTree.patterns
  }
}

class LocalFPTree(var patterns: ArrayBuffer[(String, Long)]){

  def buildHeaderTable(transactions: Iterable[ArrayBuffer[String]], minSupport: Long): ArrayBuffer[TreeNode] = {
    if(transactions.nonEmpty){
      val map: HashMap[String, TreeNode] = new HashMap[String, TreeNode]()
      for(transaction <- transactions){
        for(item <- transaction){
          if(!map.contains(item)){
            val node: TreeNode = new TreeNode(item)
            node.count = 1
            map(item) = node
          }else{
            map(item).count += 1
          }
        }
      }
      val headerTable = new ArrayBuffer[TreeNode]()
      map.filter(_._2.count >= minSupport).values.toArray.sortWith(_.count > _.count).copyToBuffer(headerTable)
      headerTable //return headerTable
    }else{
      null //if transactions is empty, return null
    }
  } //end of buildHeaderTable

  def buildLocalFPTree(
      transactions: Iterable[ArrayBuffer[String]],
      headerTable: ArrayBuffer[TreeNode]): TreeNode = {
    val root: TreeNode = new TreeNode()
    for(transaction <- transactions){
      val sortedTransaction = sortByHeaderTable(transaction, headerTable)
      var subTreeRoot: TreeNode = root
      var tmpRoot: TreeNode = null
      if(root.children.nonEmpty){
        while(sortedTransaction.nonEmpty && subTreeRoot.findChild(sortedTransaction.head.toString) != null){
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

    def sortByHeaderTable(
        transaction: ArrayBuffer[String],
        headerTable: ArrayBuffer[TreeNode]): ArrayBuffer[String] = {
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

        var cond = true //for breaking out of while loop
        var index: Int = 0

        while(cond && index < headerTable.length){
          var node = headerTable(index)
          if(node.name.equals(name)){
            while(node.nextHomonym != null)
              node = node.nextHomonym
            node.nextHomonym  = leaf
            cond = false
          }
          index += 1
        }

        addNodes(leaf, transaction, headerTable)
      }
    } //end of addNodes

    root //return root
  } //end of buildTransactionTree

  def FPGrowth(transactions: Iterable[ArrayBuffer[String]], prefix: ArrayBuffer[String], minSupport: Long){
    val headerTable: ArrayBuffer[TreeNode] = buildHeaderTable(transactions, minSupport)

    val treeRoot = buildLocalFPTree(transactions, headerTable)

    if(treeRoot.children.nonEmpty){
      if(prefix.nonEmpty){
        for(node <- headerTable){
          var result: String = ""
          val temp = new ArrayBuffer[String]()
          temp += node.name
          for(pattern <- prefix){
            temp += pattern.toString
          }
          result += temp.sortWith(_ < _).mkString(" ").toString
          patterns += result -> node.count
        }

      }

      for (node: TreeNode <- headerTable) {
        val newPostPattern: ArrayBuffer[String] = new ArrayBuffer[String]()
        newPostPattern += node.name
        if (prefix.nonEmpty)
          newPostPattern ++= prefix
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

    } //end of if
  } //end of FPGrowth
} //end of LocalFPTree