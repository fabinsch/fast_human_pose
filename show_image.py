import cv2


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))


# Start of the main program here
if __name__ == "__main__":

    # img = cv2.imread('/home/fabian/Desktop/037454012.jpg')
    img = cv2.imread('/home/fabian/Desktop/015601864.jpg')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print(mouseX, mouseY)


