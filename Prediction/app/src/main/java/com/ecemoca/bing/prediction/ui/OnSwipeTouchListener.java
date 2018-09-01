package com.ecemoca.bing.prediction.ui;

import android.content.Context;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;

import static android.view.View.Y;

/**
 * Created by bing on 1/2/18.
 */

public class OnSwipeTouchListener implements View.OnTouchListener {
    private final GestureDetector gestureDetector;

    protected OnSwipeTouchListener(Context context) {
        gestureDetector = new GestureDetector(context, new GestureListener());
    }

    public void onSwipeLeft() {

    }

    public void onSwipeRight() {

    }

    public boolean onTouch(View v, MotionEvent event) {
        return gestureDetector.onTouchEvent(event);
    }

    private final class GestureListener extends GestureDetector.SimpleOnGestureListener {
        private static final int SWIPE_DISTANCE_THRESHOLD = 100;
        private static final int SWIPE_VELOCITY_THRESHOLD = 100;

        @Override
        public boolean onDown(MotionEvent e) {
            return true;
        }

        @Override
        public boolean onFling(MotionEvent e1, MotionEvent e2, float vx, float vy) {
            float dx = e2.getX() - e1.getX();
            float dy = e2.getY() - e1.getY();

            if(Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > SWIPE_DISTANCE_THRESHOLD
                    && Math.abs(vx) > SWIPE_VELOCITY_THRESHOLD) {
                if (dx > 0)
                    onSwipeRight();
                else
                    onSwipeLeft();
            }
            return false;
        }
    }
}
