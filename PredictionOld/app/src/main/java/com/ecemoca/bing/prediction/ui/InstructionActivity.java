package com.ecemoca.bing.prediction.ui;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Toast;

import com.ecemoca.bing.prediction.R;

public class InstructionActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_instruction);

        OnSwipeTouchListener onSwipeTouchListener = new OnSwipeTouchListener(this) {
            @Override
            public void onSwipeRight() {
                 onBackPressed();
            }
        };

        findViewById(R.id.instructionScroll).setOnTouchListener(onSwipeTouchListener);

    }
}
