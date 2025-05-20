import textgrid



class TranscriptManager():


    def __init__(self, transcript_file: str):
        
        self.transcript_file = transcript_file


    def get_segment(self, start_time, end_time):

       words_in_segment = self.extract_range(self.transcript_file, start_time, end_time)

       return words_in_segment


    def extract_range(self, transcript_file, start_time, end_time):
        
        textgrid_obj = textgrid.TextGrid.fromFile(transcript_file)
        tiers = textgrid_obj.tiers

        words_in_tiers = []

        for tier in tiers:

            before, within, after = self.extract_range_tier(tier, start_time, end_time)

            words_in_tiers.append({"before": before, "within": within, "after": after})

        return words_in_tiers

    
    def extract_range_tier(self, tier, start_time, end_time):

        # words which start before the time but that end after it
        # words completely within the range
        # words which start before the range but which end after it 
        before, within, after = [], [], []

        # for all words
        for interval in tier:

            word, word_start, word_end = interval.mark, interval.minTime, interval.maxTime

            if word == '':
                continue
            
            if word_start < start_time and word_end > start_time:
                before.append((word, word_start, word_end))

            elif word_start > start_time and word_end < end_time:
                within.append((word, word_start, word_end))

            elif word_start < end_time and word_end > end_time:
                after.append((word, word_start, word_end))

            if word_end > end_time and word_start > end_time:
                break
  
        return before, within, after


if __name__ == "__main__":

    textgrid_file = "../turn-taking-projects/corpora/switchboard/switchboard_speechmatics/sw2001.TextGrid"

    textgrid_manager = TranscriptManager(textgrid_file)
    words = textgrid_manager.get_segment(0, 11)

    x=1
