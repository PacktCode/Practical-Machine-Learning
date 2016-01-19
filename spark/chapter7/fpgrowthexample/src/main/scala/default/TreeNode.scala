// Practical Machine learning
// Association rule based learning - FPGrowth example
// Chapter 7

package default

import scala.collection.mutable.ArrayBuffer

/**
 * TreeNode.scala
 * Description: This is the definition of TreeNode of FP-Tree
 * Author: Lin, Chen
 * E-mail: chlin.ecnu@gmail.com
 * Version: 1.0
 */

class TreeNode (val name: String = null, var count: Long = 0, var parent: TreeNode = null, val children: ArrayBuffer[TreeNode] = new ArrayBuffer[TreeNode](),  var nextHomonym: TreeNode = null){
  def findChild(name: String): TreeNode = {
    children.find(_.name == name) match {
      case Some(node) => node
      case None => null
    }
  }
}
