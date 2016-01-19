// Practical Machine learning
// Neural Network example
// Chapter 11

package default.algo

import org.apache.log4j.Logger


trait LoggingAbility {
  val loggerName = this.getClass.getName
  lazy val logger = Logger.getLogger(loggerName.split("\\$").head)

}
