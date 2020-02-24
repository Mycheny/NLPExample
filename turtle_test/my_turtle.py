import time
import turtle


if __name__ == '__main__':
    myPen = turtle.Pen()
    myPen.speed(0)
    myPen.pensize(5)
    circle_pos = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    color = ["black", "red", "green"]
    continent_name = ["非洲", "美洲", "亚洲"]
    continent_pos = [(0, 0+50), (100, 0+50), (50, +50+50)]
    for i in range(3):
        myPen.penup()
        myPen.goto(circle_pos[i])
        myPen.pendown()
        myPen.pencolor(color[i])
        myPen.circle(50)
        myPen.penup()
        myPen.goto(continent_pos[i])
        myPen.pendown()
        myPen.write(continent_name[i], font=("Arial", 13))
        time.sleep(1)
    turtle.done


