class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_score_acc = None
        self.early_stop = False

    # def __call__(self, acc):
    #     score = acc
    #     if self.best_score is None:
    #         self.best_score = score
    #     elif score <= self.best_score:
    #         self.counter += 1
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = score
    #         self.best_score_acc = acc
    #         self.counter = 0

    def __call__(self, loss, acc):
        score = loss
        if self.best_score is None:
            self.best_score = score
            self.best_score_acc = acc
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score_acc = acc
            self.counter = 0
