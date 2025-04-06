import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()
        # Get the current frame and traceback object
        # Extract the line number and file name from the traceback object
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename 
        # Log the error message with the file name and line number
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
        # Log the error message with the file name and line number
        #logger.logging.error(self.error_message, exc_info=True)

'''  
if __name__=='__main__':
    try:
        logger.logging.info("Enter the try block")
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
           raise NetworkSecurityException(e,sys)
'''  
# to test the exception handling, uncomment the above code and run the script.
# The above code will raise a ZeroDivisionError and the exception will be caught by the except block.